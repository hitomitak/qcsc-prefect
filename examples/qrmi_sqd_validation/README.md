# PR #42: H₂/STO-3G LUCJ の QPY・トランスパイル検証

このローカル検証は、[PR #42](https://github.com/qiskit-community/qcsc-prefect/pull/42)
の LUCJ 回路と専用トランスパイラを、QPY を使ったノード間転送に利用できるか確認するものです。
QRMI、Slurm、QPU、Prefect Cloud、ローカルの Prefect Server は不要です。
依存パッケージのインストール後、テスト中のネットワーク接続は禁止しています。

このテストがクラスタでも通ったら、次は [H₂ LUCJ の Slurm/QRMI 実行手順](README_h2_slurm.md)
で実際の `test_heron` Target と `task_runner` の接続を確認できます。

## 確認結果

Python 3.12 / macOS arm64 で **6 passed, 2 xfailed**。

| 経路 | 結果 |
| --- | --- |
| 未加工 LUCJ → QPY 保存・復元 | 測定レジスタと理想測定確率を保持 |
| 未加工 LUCJ → PR の専用トランスパイル → QPY 保存・復元 | 通過 |
| 未加工 LUCJ → QPY 保存・復元 → PR の専用トランスパイル | 既知の失敗。2 ケースを `strict=True` の `xfail` として記録 |
| ffSIM 前処理・ゲート展開 → QPY 保存・復元 → PR の専用トランスパイル → QPY 保存・復元 | 通過 |

未加工の回路を QPY から復元すると、ffsim の独自ゲートは通常の Qiskit `Gate` に変わります。
ゲートの定義は保持されますが、Python クラスの属性は復元されません。
この環境では、PR の `PRE_INIT` 内の `MergeOrbitalRotations` が `norb` 属性を参照して
`AttributeError: 'Gate' object has no attribute 'norb'` になります。
QPY 自体の測定結果が壊れているわけではありません。

通過した回避経路では、送信側で `ffsim.qiskit.PRE_INIT` を実行してから、残った ffSIM 独自ゲートを
定義へ展開して QPY に保存します。この処理には QPU の `Target` は不要です。
復元後は PR の専用トランスパイラを変更せずに呼び出しています。
実験用ヘルパー `prepare_qpy_transport()` は `examples/qrmi_sqd_validation/pr42.py` に置き、
オフラインテストと Slurm PoC が同じ処理を使います。
本番ライブラリの API にはまだ組み込んでいません。

```text
送信側: H₂ の CCSD → LUCJ → ffSIM PRE_INIT → ffSIM ゲート展開 → QPY 保存
受信側: QPY 復元 → Target を渡して PR の専用トランスパイル → ISA 回路
```

## 実行手順

リポジトリのルートで実行します。検証用 venv を新設し、動作済みの QRMI venv はそのまま使い続けてください。
Python 3.12 と、初回インストール時のパッケージ取得用ネットワークが必要です。

```bash
# Mac: cd /Users/hitomi/Project/qcsc-prefect
# vm1: cd /shared/qcsc-prefect

python3.12 -m venv /tmp/qcsc-sqd-validation
/tmp/qcsc-sqd-validation/bin/python -m pip install \
  -r examples/qrmi_sqd_validation/requirements.txt

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/qcsc-sqd-validation/bin/python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  -q -rx --tb=short
```

期待結果は `6 passed, 2 xfailed` です。`skipped` だけの場合は検証できていません。
既存の環境で収集できるよう、追加依存がない環境ではこのファイルはスキップされます。
Mac の PySCF wheel の OpenMP 警告、Runtime のプラグイン非推奨警告は今回のテスト結果に影響しません。

既知の失敗を通常のエラーとして再現したい場合:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/qcsc-sqd-validation/bin/python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  -k raw_qpy_before_pr_transpiler --runxfail -x -q --tb=short
```

このコマンドは上記の `AttributeError` で終了コード 1 になる想定です。
依存更新などで問題が解消した場合、通常のテストでも `strict=True` により `XPASS` を失敗として扱い、
既知の制約と回避処理を見直せるようにしています。

## 検証対象と限界

- H₂、結合長 0.74 Å、STO-3G、2 空間軌道、電子数 (1, 1)、4 論理量子ビット、LUCJ 1 層。
  分子積分と CCSD 振幅は PR が使う `qcsc_workflow_utility` の関数で実際に計算します。
- `meas` レジスタ、および初期測定の `test` レジスタあり・なしの両方を確認します。
- Target は 6 量子ビットの線形結合を持つ `GenericBackendV2`。CZ のみと CZ + RZZ の 2 種類で、
  PR の RZZ 後処理の有無、レイアウト・補助量子ビットの割り当てを通します。
- 全命令が Target のゲート・量子ビット配置に対応すること、未束縛パラメータがないこと、
  古典レジスタ構成、物理配置とビット順序を考慮した厳密な理想測定確率を確認します。
  QPY 往復の誤差許容値は 1e-12、トランスパイル後は 1e-10 です。
- 専用パスの構成と最適化レベル 3 は維持し、検証時間を抑えるため SABRE の探索予算だけを減らしています。
  layout trials は 10000 → 4、swap trials は 10 → 2、max iterations は 8 → 2 です。
- `test` の初期測定は理想的な全ゼロ状態であるため、状態ベクトル計算では除いて比較します。
  ノイズ下での reset mitigation の効果や、実機側の途中測定の受付は検証していません。
- 実際の `test_heron` Target、QASM/JSON 変換、`task_runner` の入出力、Slurm、Fulqrum の固有値計算は対象外です。
  実機テストに進む前に、クラスタ上でもこのテストが同じ結果になるか確認できます。
  この段階で QPU ジョブを投げる必要はありません。

## PR の参照方法

対象は `BStar14/qcsc-prefect` のコミット
`2d435b19af4648cbbc5fbc6a27ff258c93817d1a` の
`algorithms/sqd/sqd_fulqrum/main.py` です。

`packages/qcsc-prefect-qiskit/tests/fixtures/pr42/` に元ファイルの未改変スナップショットと
出典・SHA-256 を保存しています。テストはハッシュを確認し、AST から
`CircuitParameters`、`create_ansatz_circuits`、`transpile_circuit` の 3 定義をそのまま読み込みます。
`@task` は通常の関数として実行し、ロガーだけ差し替え、Prefect のサーバー起動や
Fulqrum など検証対象外の依存を避けています。分子積分ユーティリティも `.fn` で直接実行します。
PR のマージ・チェックアウト・リモートへの変更は行っていません。

依存パッケージの直接依存は `requirements.txt` に検証時の版を固定しています。
OS ごとの wheel や推移的依存まで固定したクロスプラットフォームの lockfile ではありません。

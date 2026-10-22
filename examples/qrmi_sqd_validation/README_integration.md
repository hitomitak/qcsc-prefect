# 固定PR #42のサンプリング段階をQRMIへ接続する

`integration.py` はコミット `2d435b19af4648cbbc5fbc6a27ff258c93817d1a` の
サンプリング段階をQRMI/Slurmへ接続し、後段へ渡す `BitArray` を返すローカル統合入口です。
PR本体・未改変ソースfixture・ライブラリ本体は変更していません。

```text
CircuitParameters + ElectronicProperties + options
  → 固定PRのLUCJ回路生成
  → ffSIM PRE_INIT・展開 → QPY
  → ライブラリのSlurm投入
  → 計算ノードのQRMI Target取得・固定PR専用トランスパイル
  → task_runner → PrimitiveResult
  → measの取り出し／testによるshot選別
  → BitArray
```

## 接続方法

```python
from pathlib import Path
from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime
from examples.qrmi_sqd_validation.integration import sample_bitstrings
from examples.qrmi_sqd_validation.pr42 import load_pr42

params = load_pr42().CircuitParameters(
    use_reset_mitigation=False,
    sabre_layout_trials=4,
    sabre_swap_trials=2,
    sabre_max_iterations=2,
)
runtime = QrmiSlurmRuntime(
    root_dir=Path("/shared/qrmi-pr42-bitarray/quantum"),
    command_block_name="poc-qrmi-command",
    execution_profile_block_name="poc-qrmi-execution",
    hpc_profile_block_name="poc-qrmi-hpc",
    resource="test_heron",
)
bits = await sample_bitstrings(
    circuit_params=params,
    elec_props=properties,  # 固定PRと同じElectronicProperties
    runtime=runtime,
    options={"params": {"shots": 100}},
)
assert bits.num_bits == 4
assert bits.num_shots == 100
```

runtimeにはコンパイルフックを設定せず渡します。統合入口が引数のCircuitParametersを使って
専用フックを設定したコピーを作成し、呼び出し元のruntimeは変更しません。
リソース・3 Block名・待機設定は渡されたruntimeから継承します。
既にフックやconfigがあるruntimeは、設定を黙って上書きせずエラーにします。

## 固定PRからの変更点と維持する契約

| 固定PRの処理 | ローカル統合入口 |
| --- | --- |
| `QuantumRuntime.load(runner_name)` | 構築済みのライブラリruntimeを引数で受け取る |
| `Variable.get(option_name)` | `options`を引数で受け取る。CloudのVariableを読み書きしない |
| ワークフロー側の`get_target()`と専用トランスパイル | 計算ノードのフックへ移す |
| `create_ansatz_circuits`と`transpile_circuit` | SHA-256を検証した固定PRの定義をそのまま再利用 |
| `PrimitiveResult[0].data.meas`を返す | 同じ`BitArray`型・ビット順序・shot順序を維持 |
| reset mitigationで`test.bitcount() == 0`を選別 | 対応する`meas`のshotを同じ順序で残す |

**shots省略時の既定値はPRと同じ100,000です。** ライブラリ単体の既定値100とは異なります。
この統合入口は省略時もshotsを正規化して明示的にライブラリへ渡します。
`options["params"]["shots"]`以外の設定や、bool・文字列・非正整数のshotsは受け付けません。
CLIは検証用に100 shotsを既定値とします。

reset mitigationで全shotが除外された場合は明示的な`ValueError`にします。
固定PRも空のsamplesから`BitArray`を作成できませんが、ここでは失敗理由を明確にしています。
ランダムサンプリングへのフォールバックはありません。
この関数自体にはPrefectの結果キャッシュやリトライを設定せず、ライブラリの実行Taskと
専用H₂ Flowでも自動リトライを無効にしています。

## オフライン検証

既存の検証用環境を使い、リポジトリルートで実行します。

```bash
PYTHONPATH=packages/qcsc-prefect-blocks/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  -q --tb=short
```

Slurm・QRMIサービスを代替し、実際のQPY往復、専用トランスパイル、QASM変換、
代替CLIのJSON入出力、ResultDecoderを経由して`BitArray`まで確認します。
H₂・100 shots、CZのみ／CZ+RZZ、`meas`のみ／`meas`+`test`の組み合わせを検査します。
非対称ビット列と非ゼロの`test`を含め、ビットの向き・shot対応・選別後の順序を検査します。

固定PRの`sample_bitstrings`と`_resolve_shots`も未改変ASTからテスト内で読み込み、
直接Runtimeの境界だけを代替した参照実行と、返却値を比較します。
PRのFlow全体をそのままQRMIへ差し替えて実行したという意味ではありません。
既定shots、不正options、Slurm失敗、全shot除外、H₂入口の成果物保存も検査します。

## クラスタで統合入口を実行する場合

新しい統合入口の実機実行は未確認です。以前のライブラリ単体の成功とは区別してください。
[H₂クラスタ手順](README_h2_slurm.md)で使用した同じ配置先・共有venv・Cloud workspace・
3 Blockを使えます。`examples/qrmi_sqd_validation`と更新したテストファイルを転送し、
ワークフローと計算ノードから同じライブラリ・固定PRフックをimportできることを確認します。

次のコマンドは**実際のSlurm・量子ジョブを作成します**。

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m examples.qrmi_sqd_validation.integration \
    --shots 100 --work-dir /shared/qrmi-pr42-bitarray
```

`--reset-mitigation`を付けなければ`meas`だけを要求します。
各`run_<ID>`に以下を保存します。

- `sampling_summary.json`: 固定PRの出典、要求／返却shots、幅、counts、作業先。
- `bitstrings.txt`: 返却したBitArrayの1 shot 1行のビット列。shot順序を保持。
- `quantum/<ID>/`: ライブラリのQPY・request・config・report・生のresult・Slurm記録。

`h2_bitarray_flow()`のPythonの戻り値は`BitArray`です。CLIも最後に幅・shots・countsを表示します。
保存したファイルから後段用の値を復元する場合は、summaryの幅を使用します。

```python
bits = BitArray.from_samples(
    (run_dir / "bitstrings.txt").read_text().splitlines(),
    num_bits=summary["num_bits"],
)
```

この段階はBitArray返却までです。配置回復、Fulqrum実行、エネルギーの検証は含みません。
`test`付きの実機受付・ノイズ下の効果も未確認です。

# QRMI/Slurm Sampler ライブラリ

`qcsc_prefect.integrations.qiskit.qrmi.QrmiSlurmRuntime` は、測定済み・パラメータ束縛済みの
1回路をQPYで共有領域へ保存し、Slurmの計算ノードでTargetを取得してトランスパイルし、
QRMIの `task_runner` を通してBatchで実行します。戻り値は `PrimitiveResult` です。
Qiskitの `BaseSamplerV2` の実装や `prefect_qiskit.QuantumRuntime` のPrefect Blockではありません。

## 責務の境界

| ライブラリ | アルゴリズム側 |
| --- | --- |
| QPY/JSON転送、Slurm投入・待機、Batch環境検査 | 分子積分・CCSD・LUCJ回路の生成 |
| 計算ノード内でQRMI Targetを取得 | QPY保存前のffSIM PRE_INIT・独自ゲート展開 |
| フック呼び出し、Target対応命令の検査 | PR固有のトランスパイルと固定ソースの照合 |
| QASM payload、CLI起動、PrimitiveResult復元 | reset mitigationのビット列選別、SQD、Fulqrum |
| 全古典レジスタの名前・幅・shotsの検査、診断記録 | SABRE探索予算・seedなどアルゴリズム固有の設定 |

ライブラリは `examples`、ffSIM、PySCF、PRスナップショットを参照しません。
QRMIは計算ノードのエントリポイント内でのみimportします。
executorは実行時に遅延importし、既存のnative Qiskit機能には依存を追加しません。

## 配置と基本的な使い方

ワークフローと計算ノードの両方に、同じ版の `qcsc-prefect-qiskit` をインストールします。
ワークフロー側には追加でexecutorとその依存が必要です。
ソースから個別に入れる場合は既存のworkspaceパッケージも依存として解決できる状態で実行します。

```bash
python -m pip install -e './packages/qcsc-prefect-qiskit[slurm]'
```

計算ノードでは既存のQRMI、SPANK、`task_runner` の動作環境を使用します。
CommandBlockの `executable_key` に対応するPythonでライブラリをimportできる必要があります。
新規CommandBlockの `default_args` は次のように設定できます。

```python
["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"]
```

同じPython環境にconsole script `qcsc-qrmi-sampler` も登録されます。
ExecutionProfileBlockは `launcher="single"`、HPCProfileBlockは `hpc_target="slurm"`、
`slurm_qpu` は下記 `resource` と同じIDを設定します。
共有ジョブディレクトリが両ホストの同じ絶対パスで読み書きできる必要があります。
SPANKがPython起動前にBatchを確保するよう、対象リソースの
`QRMI_IBM_QCS_SESSION_MODE="batch"` を各ノードで設定します。
ライブラリが実行中にセッションモードを変更することはありません。

```python
from pathlib import Path
from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime

runtime = QrmiSlurmRuntime(
    root_dir=Path("/shared/quantum-jobs"),
    command_block_name="my-qrmi-command",
    execution_profile_block_name="my-qrmi-execution",
    hpc_profile_block_name="my-qrmi-hpc",
    resource="test_heron",
)
result = await runtime.sampler([(circuit,)], shots=100)
# 同じ指定のPR #42形式
result = await runtime.sampler(
    sampler_pubs=[(circuit,)], options={"params": {"shots": 100}}
)
```

2回呼べば2件のジョブになります。各呼び出しは独立したディレクトリを作成し、
キャッシュ・自動リトライを無効にしています。失敗時の再呼び出しも新規投入です。
`watch_poll_interval` と `timeout_seconds` は既存executorに渡します。
待機タイムアウトは投入済み量子ジョブのキャンセルを保証しません。

## PR #42 Runtime 呼び出しとの互換性

照合対象はコミット `2d435b19af4648cbbc5fbc6a27ff258c93817d1a` の
`algorithms/sqd/sqd_fulqrum/main.py` です。ローカルの未改変スナップショットと
SHA-256を使って確認しています。

| 固定PRの処理 | 対応範囲 |
| --- | --- |
| `await QuantumRuntime.load(runner_name)` | 対象外。上記クラスを直接構築する |
| `await runtime.get_target()` → 呼び出し側で専用トランスパイル | 対象外。Target取得と専用トランスパイルを計算ノードのフックへ移す |
| `await runtime.sampler(sampler_pubs=[(circuit,)], options=options)` | `options={"params": {"shots": 正の整数}}` に対応 |
| `pub_result[0].data.meas`、追加 `test` レジスタ | 名前・幅・shotsを保持して返す。選別処理はアルゴリズム側に残す |

固定PRの **sampler呼び出し式そのもの** をASTから抽出し、新クラスに対して実行する
オフラインテストを追加しています。PRのFlow全体を無変更で置き換えられるという意味ではありません。
PR #42への変更、コメント、pushは行いません。

固定PRのサンプリング段階から後段用の`BitArray`を返すローカルアダプターは
[QRMI統合入口](../../examples/qrmi_sqd_validation/README_integration.md)にあります。
runtimeとoptionsを注入し、固定PRの回路生成・専用トランスパイルを再利用します。

明示的な `shots=` と `options["params"]["shots"]` が異なる場合はエラーです。
両方省略したときのライブラリの既定値は100 shotsです。
固定PRの `Variable.get` が設定する既定値100,000 shotsとは異なりますが、
PRと同様に `options` で明示した場合はその値を使用します。
文字列、浮動小数、boolはshotsに受け付けません。
他のoptions、複数PUB、PUB内のパラメータ値・shots、未束縛回路は投入前にエラーにします。
Runtimeのmitigation等の指定を黙って落とすことはありません。

## アルゴリズム固有フック

`prepare(circuit) -> PreparedCircuit` はワークフロー側でQPY保存前に実行します。
回路のコピーを渡し、戻り値の `config` をJSONとして保存します。
`transpiler="module:function"` は計算ノードで
`function(restored_circuit, qrmi_target, config) -> QuantumCircuit` として呼びます。
両フックとも古典レジスタの名前と幅を保持する必要があります。
フックのモジュールは計算ノードからimportできるよう配置してください。
ジョブ内のJSONはワークフローが生成する信頼済みファイルとして扱います。

既定フックは最適化レベル1のgeneric transpilerで、アルゴリズム固有configは受け付けません。
カスタム `transpile_config` と `PreparedCircuit.config` が重なる場合は後者を優先します。

SQD用の接続例は次のとおりです。

```python
from functools import partial
from examples.qrmi_sqd_validation.pr42 import prepare_sampler_circuit

runtime = QrmiSlurmRuntime(
    root_dir=Path("/shared/quantum-jobs"),
    command_block_name="poc-qrmi-command",
    execution_profile_block_name="poc-qrmi-execution",
    hpc_profile_block_name="poc-qrmi-hpc",
    resource="test_heron",
    prepare=partial(prepare_sampler_circuit, circuit_parameters=params.model_dump()),
    transpiler="examples.qrmi_sqd_validation.pr42:transpile_for_target",
)
result = await runtime.sampler([(lucj_circuit,)], options={"params": {"shots": 100}})
```

ffSIM前処理・展開は専用フックが行います。未加工LUCJをQPYで往復するとffSIM属性を失う
既知の制約は残っています。ライブラリが任意の独自ゲートのPython属性を復元するわけではありません。

## PoCからの移行

既存 `examples/prefect_bitcount_demo/qrmi_slurm_runtime.py` は旧コンストラクタと
`poc-qrmi-*` Block名を保つ薄い互換層です。`transpile_strategy="sqd-pr42"` を
上記フックへ変換します。H₂ FlowとBitCount Flowの呼び出し方は維持しています。
既存の `qrmi_task_runner.py` もライブラリのworkerへ転送するため、CommandBlockの
スクリプトパスを変更する必要はありません。

**配置時はワークフローと計算ノードの両方を更新してください。** 新しいsenderは
`--request sampler_request.json` を渡すため、旧workerとの混在は未対応です。
共有ソースだけをrsyncした場合も、両側で更新したパッケージをインストールするか、
既にeditable installされていて更新ソースを参照していることを確認します。
SQDフックは引き続き検証用example・固定PR fixture・化学ユーティリティを必要とします。

各ジョブには `circuit.qpy`、`sampler_request.json`（schema version 1）、
`transpile_config.json`、`sampler.slurm`、`isa.qpy`、`transpile_report.json`、
`sampler_input.json`、生の `result.json` を保存します。
executorが終了結果を返した場合は、失敗状態でも `slurm_job.json` を残します。
workerとsenderの両方で1 PUB・全レジスタの名前、幅、shotsを検査します。
CLIが終了コード0でも不正な結果なら失敗として扱います。

## オフライン検証と未確認範囲

検証用環境にQiskit/Runtime/Prefectとexampleの追加依存がある状態で実行します。
以下のテストではTaskの `.fn` を使い、Slurm/QRMIを代替します。

```bash
PYTHONPATH=packages/qcsc-prefect-blocks/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_slurm_runtime.py \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_slurm_bitcount_poc.py -q
```

以前のPoCではH₂/STO-3G、4量子ビットLUCJ、`test_heron`、100 shots、4ビット`meas`の
クラスタ実行が確認されています。ライブラリ化後も、利用者から以下の成功報告を受けています。

- Slurm job: `16`、Compute host: `slurm-vm1`。
- H₂/STO-3G、100 shots、4ビット`meas`のみ。カウント合計100。
- 証跡ディレクトリ:
  `/shared/qrmi-h2-library-check.UJ8xXq/runs/run_04c127d0c7d148229c809de682475227/quantum/6ed4fa77b0714d9083a701c9d2351949`。

追加したBitArray統合入口の実機実行、`test`レジスタ付きの実機受付、
ノイズ下のreset mitigation、Fulqrum全体の実行は未確認です。

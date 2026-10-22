# H₂ LUCJ を既存 Slurm/QRMI PoC で実行する

オフライン検証の次に、実際の `test_heron` Target と `task_runner` の接続を確認する手順です。
既存の BitCount PoC が動いたクラスタ設定、Prefect Cloud 接続、量子用の `poc-qrmi-*` 3 Block を使います。
この Flow は H₂ の量子サンプリングまでを実行します。Fulqrum によるエネルギー計算は次の段階です。

```text
vm1 Workflow: H₂/STO-3G の CCSD → PR #42 の LUCJ 回路生成
  runtime.sampler([(circuit,)], shots=100)
    → ffSIM PRE_INIT・ゲート展開 → circuit.qpy / transpile_config.json 保存
    → 既存 Slurm Integration → sbatch --qpu=test_heron
Compute Node: QPY 復元 → QRMIService から Target 取得
    → PR #42 の専用トランスパイル → isa.qpy / transpile_report.json 保存
    → task_runner → IBM Quantum Batch → result.json
vm1 Workflow: PrimitiveResult 復元 → meas の4ビット・shots数確認 → summary.json
```

利用者の Flow が呼ぶ量子実行 API は `sampler()` です。
内部の Compute Node では Target 取得に `QRMIService`、送信に `task_runner` を使います。
PR の関数は固定コミットのスナップショットから読み込みます。リモート PR は変更しません。

## 1. 更新ファイルを共有領域へ配置

Mac のリポジトリのルートで実行します。`VM1_SSH_HOST` は普段の接続先に置き換えます。

```bash
rsync -avR --exclude='__pycache__/' --exclude='.pytest_cache/' \
  examples/qrmi_sqd_validation \
  examples/prefect_bitcount_demo/qrmi_slurm_runtime.py \
  examples/prefect_bitcount_demo/qrmi_task_runner.py \
  algorithms/qcsc_workflow_utility/src \
  packages/qcsc-prefect-qiskit/tests/fixtures/pr42 \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  hitomi@VM1_SSH_HOST:/shared/qcsc-prefect/
```

元の BitCount 用 rsync コマンドだけでは、新しい example、化学ユーティリティ、
PR のスナップショットが転送されません。上の一式が必要です。
`/shared/qcsc-prefect` は両 Compute Node から同じパスで読める前提です。

## 2. 実行する QRMI 環境の追加依存を確認

以降は vm1 で実行します。既存の Slurm Block は `/shared/venv/qrmi/bin/python` を起動するため、
`/tmp/qcsc-sqd-validation` に依存を入れただけでは Compute Node の実行環境には反映されません。
追加依存は ffSIM、PySCF、pydantic-numpy と検証用の pytest です。

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate

python -m pip install --dry-run \
  'ffsim==0.0.84' 'pyscf==2.13.0' 'pydantic-numpy==9.1.0' 'pytest==9.0.3'
```

表示された依存差分を確認してから、追加インストールします。
NumPy、Pydantic などの依存も更新される場合があるため、前後の環境を記録します。
QRMI や SPANK のソース・設定の変更は不要です。

```bash
python -m pip freeze > /shared/qrmi-before-sqd-poc.txt
python -m pip install 'ffsim==0.0.84' 'pyscf==2.13.0' 'pydantic-numpy==9.1.0' 'pytest==9.0.3'
python -m pip check
python -m pip freeze > /shared/qrmi-after-sqd-poc.txt

python - <<'PY'
from importlib.metadata import version
from qrmi import QRMIService
from examples.qrmi_sqd_validation.pr42 import load_pr42
load_pr42()
for name in ('qrmi', 'qiskit', 'qiskit-ibm-runtime', 'ffsim', 'pyscf', 'pydantic-numpy'):
    print(f'{name}=={version(name)}')
print('QRMI import and pinned PR #42 loader OK')
PY
```

この import 確認は QPU を確保しません。Python 3.12 を前提とし、Qiskit 2.4.0、
qiskit-ibm-runtime 0.46.1、ffsim 0.0.84 の組み合わせをローカル検証しています。
クラスタの QRMI 環境でも、次のオフラインテストを通してから量子ジョブへ進みます。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_sqd_qpy_transpile.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_sqd_slurm_poc.py \
  -q --tb=short
```

期待結果は **13 passed, 2 xfailed**。
新しい7テストは Slurm/QRMI を代替し、QPY → 専用トランスパイル → 実際の QASM 変換・復元 →
CLI 結果 JSON の受け渡しを確認します。実機ジョブは投入しません。
送信側・Compute Node 側の参照コミットが異なる場合に、CLI 起動前に停止することも確認します。

## 3. Prefect Cloud 接続と既存 Block を確認

BitCount PoC と同じターミナル設定・Cloud workspace を使います。

```bash
prefect config view
python - <<'PY'
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
command = CommandBlock.load('poc-qrmi-command')
execution = ExecutionProfileBlock.load('poc-qrmi-execution')
hpc = HPCProfileBlock.load('poc-qrmi-hpc')
assert hpc.slurm_qpu == 'test_heron'
assert hpc.executable_map['python'] == '/shared/venv/qrmi/bin/python'
assert execution.launcher == 'single'
print('Command:', command.default_args)
print('Existing QRMI Slurm blocks OK')
PY
```

`Command` が共有ソースの `examples/prefect_bitcount_demo/qrmi_task_runner.py` を指すことを確認します。
同じパスへ更新ファイルを配置した場合、Block の再登録は不要です。
Cloud 接続や Block が未設定の場合は、[既存 PoC の手順](../prefect_bitcount_demo/README_qrmi_slurm.md)
の Cloud 接続・Block 登録を先に行います。

## 4. H₂ LUCJ を100 shotsで実行

ここから先は実際の Slurm・量子ジョブを1件ずつ作ります。
初回は `reset_mitigation=False`、`meas` レジスタだけで確認します。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m examples.qrmi_sqd_validation.flow \
  --shots 100 --work-dir /shared/qrmi-h2-poc
```

Flow 内の呼び出しは以下です。

```python
runtime = QrmiSlurmRuntime(
    root_dir=run_dir / 'quantum',
    transpile_strategy='sqd-pr42',
    sqd_circuit_parameters=params.model_dump(),
)
result = await runtime.sampler([(circuit,)], shots=shots)
```

H₂ は結合長 0.74 Å、STO-3G、4 論理量子ビット、LUCJ 1層です。
専用トランスパイルの最適化レベルは3、SABREの探索予算はオフライン検証と同じ
layout=4 / swap=2 / iterations=2、seed=24です。
自動リトライは無効で、再実行するたび新しいジョブを作ります。

## 5. 出力の確認

Flow のログと最後の JSON に `work_dir` が出ます。
量子ジョブの作業先は `summary.json` の `slurm.work_dir` です。

```text
/shared/qrmi-h2-poc/run_<ID>/
  summary.json
  quantum/<ID>/
    circuit.qpy             # ffSIM 前処理・展開済みの転送回路
    transpile_config.json   # 固定PRの出典、パラメータ、seed
    sampler.slurm
    output.out / output.err
    isa.qpy                 # 実際の Target に対する専用トランスパイル結果
    transpile_report.json   # 入出力の演算、Target幅、レジスタ、host、mode
    sampler_input.json      # task_runner への実際の入力
    result.json             # task_runner の生出力
    slurm_job.json
```

成功条件は、Slurm が `COMPLETED`、専用トランスパイル済み命令が Target に適合し、
要求した100 shotsの4ビット `meas` が返ることです。
`summary.json` の `quantum_mode: batch`、`transpile_strategy: sqd-pr42`、
`registers: {"meas": 4}`、カウントの合計100を確認します。
IBM 側でも `output.out` の Task ID を使って Batch に入ったことを確認できます。
`test_heron` の返却分布が理想的な H₂ 分布と一致することは、この接続確認の成功条件には含めません。

失敗時は同じジョブを投入し直す前に `output.err` と `transpile_report.json` の有無を確認します。
`norb` のエラーなら更新前の送信コードや未加工QPYを使っていないか、
`ModuleNotFoundError` なら Slurm が起動する `/shared/venv/qrmi/bin/python` の追加依存を確認します。
PR source mismatch なら、両側から同じ共有ソース・スナップショットを参照できているか確認します。
Batch 設定・SPANK・scontrol 周りは既存 BitCount PoC と共通です。

`meas` の実行が通った後の別テストとして、`--reset-mitigation` を付けると
初期測定の `test` レジスタも4ビット・同じshots数で返るか確認できます。
その実機受付とノイズ下の効果は、オフライン検証だけでは確認できません。

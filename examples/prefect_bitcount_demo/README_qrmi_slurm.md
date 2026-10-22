# QRMI Batch + Slurm MPI BitCount PoC

QRMI/Slurm実行は `qcsc-prefect-qiskit` のライブラリへ移動しました。
exampleのコンストラクタ・Block名・workerのスクリプトパスは維持しています。
[移行手順](../../docs/howto/howto_qrmi_slurm_sampler.md#pocからの移行)に従い、
ワークフローと計算ノードの両側でパッケージを更新してください。
以下の従来のrsync対象だけでは、新しいライブラリは転送されません。

`test_heron`、`debug` partition、共有領域 `/shared`、既存のQRMI仮想環境
`/shared/venv/qrmi` を使う最小デモです。Workflowはvm1上のPythonで実行します。

```text
vm1: Prefect Workflow
  runtime.sampler([(circuit,)], shots=100)
    -> 既存 run_job_from_blocks() -> sbatch --qpu=test_heron
       Compute Node: circuit.qpyを読み、Target取得・トランスパイル
         -> task_runner test_heron sampler_input.json result.json
         -> IBM Quantum Batch -> 結果を /shared へ保存
    <- PrimitiveResultとして復元
  測定ビット列をinput.binに書く
    -> 既存BitCountタスク -> sbatch（QPU指定なし）
       2 Compute Nodes: mpirun get_counts_hist
    <- hist_u64.binを読む
  元の測定ビット列とMPIの集計結果が完全一致することを確認
```

`QrmiSlurmRuntime`は、このPoC用に`sampler()`を持たせた薄いクラスです。
`prefect_qiskit.QuantumRuntime`の差し替えや登録は行いません。
受け付ける入力は、測定付き・パラメータ束縛済みの回路1個だけです。
回路は既存BitCountプログラムに合わせた10量子ビットのGHZ回路です。
量子回路の送信は実際のQRMI `task_runner`コマンドが担当します。

## 1. ブランチをクラスタへ配置

このブランチの未コミット変更も含めて、必要なソースをvm1の
`/shared/qcsc-prefect`へ配置します。Mac上のリポジトリのルートから実行します。
`VM1_SSH_HOST`は、普段vm1へ接続するときのSSH接続先に置き換えてください。

```bash
ssh hitomi@VM1_SSH_HOST 'mkdir -p /shared/qcsc-prefect'
rsync -avR --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.env' \
  packages/qcsc-prefect-core \
  packages/qcsc-prefect-blocks \
  packages/qcsc-prefect-adapters \
  packages/qcsc-prefect-executor \
  examples/prefect_bitcount_demo \
  hitomi@VM1_SSH_HOST:/shared/qcsc-prefect/
```

以降は、特記のない限りvm1で`hitomi`として実行します。
`/shared`はvm2からも同じパスで参照できる前提です。

## 2. SPANKの量子実行モードをBatchに変更

**vm1とvm2の両方**の`/etc/slurm/qrmi_config.json`で、既存の`test_heron`の
`environment`に次の1項目を追加します。他の接続情報はそのまま使います。

```json
"QRMI_IBM_QCS_SESSION_MODE": "batch"
```

編集例（各ノードで実行）:

```bash
sudo cp -p /etc/slurm/qrmi_config.json /etc/slurm/qrmi_config.json.before-batch
sudo vi /etc/slurm/qrmi_config.json
```

元のファイルの`root:root`、モード`600`を維持します。設定は新しいジョブの
開始時に読み込まれるので、この設定変更のためのSlurm再起動は不要です。
既存の実行枠のモードは変わりません。以後に投入するジョブで確認します。

SPANKがPython開始前にBatchを確保し、そのIDを
`test_heron_QRMI_JOB_ACQUISITION_TOKEN`として渡します。
Batchでも環境変数やAPIには`SESSION`という語が使われます。
Python側でモードを後からexportしても、取得済みの実行枠は切り替わりません。
PoCは渡されたモードが`batch`でない場合、量子ジョブを送らず失敗します。

設定を値の漏えいなく確認するには、各ノードで次を実行します。

```bash
sudo python3 - <<'PY'
import json
with open('/etc/slurm/qrmi_config.json') as stream:
    resource = next(r for r in json.load(stream)['resources'] if r['name'] == 'test_heron')
assert resource['environment']['QRMI_IBM_QCS_SESSION_MODE'] == 'batch'
print('test_heron: batch configured')
PY
```

QRMIの[モードと取得済みIDの扱い](https://github.com/qiskit-community/qrmi/blob/main/src/ibm/quantum_compute_service.rs)、
SPANKの[設定読み込みとリソース確保](https://github.com/qiskit-community/spank-plugins/blob/main/plugins/spank_qrmi/spank_qrmi.c)
に基づく設定です。

## 3. 既存環境へPrefectとローカルパッケージを追加

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate
python -m pip install 'prefect==3.6.27' 'fastapi==0.136.0' \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-executor
```

既に動いている`qrmi[ibm]`とその依存パッケージを使います。
制御側にも`qiskit`と`qiskit-ibm-runtime`が必要なので、ここでは同じ共有venvを
使います。`prefect-qiskit`のインストールは不要です。
このPoCではPrefect 3.6.27とFastAPI 0.136.0を固定します。
FastAPI 0.155以降は`Body()`の未定義キーワード引数を受け付けず、
このPrefectのServerコードで`Body() got an unexpected keyword argument 'min_items'`
になります。Serverのimportと`/api/health`は固定した組み合わせで検証済みです。
ローカルの量子ライブラリの検証バージョンはQiskit 2.4.0、
qiskit-ibm-runtime 0.46.1です。既存のQRMI環境はそのまま使い、結合は現地で確認します。

```bash
command -v sbatch scontrol scancel
/shared/venv/qrmi/bin/task_runner --help
/shared/venv/qrmi/bin/python -c 'from importlib.metadata import version; print({p: version(p) for p in ("qrmi", "qiskit", "qiskit-ibm-runtime", "prefect", "fastapi")})'
/shared/venv/qrmi/bin/python -c 'import prefect.server.api.server; print("Prefect Server import OK")'
```

新しいターミナルでは、作業を続ける前に
`source /shared/venv/qrmi/bin/activate`を実行してください。
`/usr/lib64/python3.9`から`PackageNotFoundError: qrmi`が出る場合は、仮想環境外の
Pythonを実行しています。上の確認コマンドはPythonも絶対パスで指定しています。

このブランチでは単一ジョブを`scontrol`で追跡するので、Slurm accountingや
SlurmDBDの有効化は不要です。過去ジョブを後から復旧する機能は含みません。
Workflowを止めず、投入直後から完了まで追跡します。

## 4. 既存MPI BitCountをビルド

```bash
cd /shared/qcsc-prefect
source /etc/profile.d/modules.sh
module unload mpi/openmpi-x86_64
module load mpi/openmpi-x86_64
mkdir -p examples/prefect_bitcount_demo/bin
mpicxx -O3 -std=c++17 \
  -o examples/prefect_bitcount_demo/bin/get_counts_hist \
  examples/prefect_bitcount_demo/src/get_counts_hist.cpp
```

既存のMPIプログラムをそのまま使います。2ノード・各1プロセスで起動し、
クラスタで動作確認済みの`eth0`を使うOpen MPI設定を適用します。

## 5. Prefect Cloudへ接続し、PoC用Blockを作成

この環境ではPrefect Cloudをそのまま使います。ローカルServerを起動する必要は
ありません。Cloudは状態・ログ・Block・Artifactsを管理し、WorkflowのPythonは
vm1、量子とMPIの実行プログラムはSlurmのCompute Nodeで動きます。
このPoCはPythonから直接Flowを起動するため、Prefect Workerの起動も不要です。

vm1で、Workflowを起動するターミナルからCloudへログインします。
以前の案内でlocalhostの`PREFECT_API_URL`をexportしていた場合は解除します。
環境変数はprofileの設定より優先されるためです。

```bash
cd /shared/qcsc-prefect
source /shared/venv/qrmi/bin/activate
unset PREFECT_API_URL
prefect cloud login
prefect config view
```

普段使うCloudのworkspaceを選びます。APIキーを使う場合はCLIの対話入力に
貼り付けます。ブラウザへのログインだけではvm1のPythonの接続設定はされません。
設定の`PREFECT_API_URL`が`https://api.prefect.cloud/api/accounts/.../workspaces/...`
になっていることを確認します。既存のCloud用profileを使っても構いません。

接続確認には、Cloudの認証も扱うPrefectクライアントを使います。

```bash
python - <<'PY'
import asyncio
from prefect.client.orchestration import get_client
from prefect.settings import PREFECT_API_URL

assert PREFECT_API_URL.value(), 'Configure the Prefect Cloud workspace first'
async def check():
    async with get_client() as client:
        error = await client.api_healthcheck()
        if error:
            raise error
    print('Prefect API connection OK')
asyncio.run(check())
PY
python -m examples.prefect_bitcount_demo.create_qrmi_slurm_blocks
```

`PREFECT_API_URL`未設定ではPrefectが一時Serverを起動しようとします。
Block登録もFlow実行も、同じCloud workspaceへの接続設定をしてから行います。
実行状況は通常のPrefect Cloud UIで確認します。SSHトンネルは不要です。
詳しいログイン方法は[Prefect公式ドキュメント](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud)
を参照してください。

### 任意: ローカルServerを使う場合だけ

Cloudを使わない場合は、vm1の別ターミナルで同じvenvを有効にして
`prefect server start --host 127.0.0.1`を実行します。Workflowのターミナルで
接続先を設定してから、上の接続確認とBlock登録を行います。

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

MacからローカルServerのUIを見るときだけ、
`ssh -N -L 4200:127.0.0.1:4200 hitomi@VM1_SSH_HOST`でトンネルを開き、
ブラウザで`http://127.0.0.1:4200`を開きます。

作成・更新されるのは、このPoC専用の6個のBlockだけです。

| 用途 | Command | Execution Profile | HPC Profile |
|---|---|---|---|
| QRMI | `poc-qrmi-command` | `poc-qrmi-execution` | `poc-qrmi-hpc` |
| MPI | `poc-bitcount-command` | `poc-bitcount-execution` | `poc-bitcount-hpc` |

量子側は動作済み`run_bell.sbatch`と同じく、batch allocation内でPythonを
直接起動します（`launcher="single"`）。そのPythonが`task_runner`を起動します。
MPI側は`launcher="mpirun"`です。MPI用HPC Profileには`slurm_qpu`を指定しません。

## 6. Workflowを実行

```bash
python -m examples.prefect_bitcount_demo.flow_qrmi_slurm --shots 100
```

デフォルトの作業先は`/shared/qrmi-bitcount-poc/run_<ID>/`です。
変更する場合は`--work-dir /shared/別のディレクトリ`を渡します。
Workflowには2つのPrefect Taskと2つのSlurmジョブが順に現れます。

```text
quantum-sampling-qrmi-slurm  -> 1ノード、--qpu=test_heron、Batch
hpc-bitcount-task           -> 2ノード、mpirun、QPU指定なし
```

成功すると、量子側とMPI側の別々のSlurm Job ID、`counts_match: true`、
100ショット分の集計を出力します。PrefectのArtifactsにも量子・MPIの
カウント比較テーブルと各Slurmジョブの情報が出ます。
出力された作業先を使って確認できます。

```bash
cat /shared/qrmi-bitcount-poc/run_<ID>/summary.json
cat /shared/qrmi-bitcount-poc/run_<ID>/bitcount/output.out
```

MPIログの`--display-map`出力で2ノードへのプロセス配置を確認します。
量子側の`quantum/<ID>/output.out`にはホスト、`mode: batch`、ISA回路の演算と
測定結果、`task_runner`のTask IDが出ます。IBMのポータルでも、その量子ジョブが
Batchに含まれることを確認してください。

`test_heron`ではGHZの理想分布を成功条件にしません。今回の成功条件は、
要求ショット数が返り、その全ビット列を別のMPIジョブで集計した結果が一致することです。

## ファイルとトラブルシュート

| ファイル | 役割 |
|---|---|
| `flow_qrmi_slurm.py` | 量子とMPIを順に実行し、結果を比較 |
| `qrmi_slurm_runtime.py` | `sampler()`から既存Slurm Integrationを呼ぶ |
| `qrmi_task_runner.py` | Compute NodeでトランスパイルしCLIを実行 |
| `create_qrmi_slurm_blocks.py` | クラスタ用の固定設定を登録 |

- モード設定エラー: 両ノードの`test_heron.environment`を確認します。
- `task_runner`失敗: `quantum/<ID>/output.out`と`output.err`を確認します。
  終了コード0でも結果が欠けていればPoCは失敗します。
- `task_runner returned an unexpected sampler result`（旧コード）:
  入力の`support_qiskit=False`と、`PrimitiveResult`を期待する受け取り側が
  不整合でした。更新版の`qrmi_task_runner.py`を共有ソースへ配置して再実行します。
  更新版はQRMI標準Samplerと同じ`support_qiskit=True`を使い、PUBにもshotsを
  明示します。Block再登録やSlurm再起動は不要です。更新版では結果型、PUB数、
  shots数のどれが想定と異なるかと、生の`result.json`の保存先をエラーに出します。
- MPI待ち: `squeue`で2ノードが確保できるか確認します。
- `drained / batch job complete failure`かつジョブのログがない:
  Compute Nodeの`/var/log/slurm/slurmd.log`を確認します。
  `required plugin spank_qrmi.so: init_post_opt() failed`なら、MPI本体の実行前に
  SPANKが失敗しています。旧版には`--qpu`未指定でも`SLURM_ERROR`を返す不具合が
  あり、[upstreamで修正されています](https://github.com/qiskit-community/spank-plugins/commit/a61aecf1f04a2936e5ebfa32c84bf0207364b66d)。
  ソースを変更しないPoCの回避策は、後述の`plugstack.conf`での`optional`指定です。
  MPIジョブにQPU指定を追加する必要はありません。
  Slurm内部の起動失敗では終了コード0が返る場合があるため、このPoCはSlurmの
  最終状態も確認します。`idle*`の`*`は応答未確認を表し、解除直後は再登録を
  待ちます。`*`が残る場合はslurmdの状態・登録ログを確認します。
- MPI起動エラー: 両ノードのOpen MPI、module、`eth0`設定を確認します。
- `mpirun: command not found`: moduleのロード済み情報とPATHが食い違っている
  可能性があります。ジョブ内でOpen MPIをunloadしてからloadし直し、
  `command -v mpirun`で起動前に確認します。更新版のBlock作成コードには
  この順序を入れています。配置後に
  `python -m examples.prefect_bitcount_demo.create_qrmi_slurm_blocks`で再登録します。
- `hist_u64.bin`がない: `bitcount/bitcount.slurm`の末尾と`output.out`、
  `output.err`を確認します。起動行は
  `mpirun --display-map "${QCSC_PREFECT_EXECUTABLE}"`です。
  `mpirun --display-map`だけなら実行ファイルが欠けています。
  クラスタ側で読み込まれているadaptersのSlurmテンプレートを、このブランチと
  揃えてください。既存の`input.bin`を使って、修正したスクリプトを
  `sbatch --wait bitcount.slurm`で再投入すれば、量子ジョブを再実行せずMPIだけ
  確認できます。手動再投入では失敗済みFlowの状態や`summary.json`は更新されません。
- `sacct`のエラー: このブランチのadaptersをeditable installしたか確認します。
- `Body() ... 'min_items'`: 共有venvで
  `python -m pip install 'prefect==3.6.27' 'fastapi==0.136.0'`を実行します。
  Cloud利用時は`prefect cloud login`で接続先を設定し、上のクライアントで
  接続確認してからBlock登録をやり直します。CloudのServerを起動・再起動する
  必要はありません。ローカルServerを使っている場合だけ再起動します。
- `Invalid job id`: controllerが終了済みジョブを削除した後の復旧は対象外です。
  作業ファイルとログを残し、新しいFlow Runでやり直します。

自動再投入は量子タスクでは無効です。再実行すると新しいSlurm・量子ジョブが
作られます。APIキーや取得済みIDの値をデモコードに記載する必要はありません。

結果形式の指定は[QRMIのSampler実装](https://github.com/qiskit-community/qrmi/blob/main/python/qrmi/primitives/base_sampler.py)
に合わせています。`support_qiskit=False`のJSONは、`ResultDecoder.decode()`を
通すだけでは`PrimitiveResult`になりません。shotsはトップレベルでも指定可能ですが、
このPoCでは標準実装と同じくPUBの3番目にも渡します。

## ソースを変更せず旧SPANKプラグイン（11b4b8e）を回避する

MPI側も既存のSlurm Integrationから`sbatch`を呼びます。
`--qpu`を指定しないMPIジョブはQRMI APIや`task_runner`を呼びませんが、
Slurmに登録した`spank_qrmi.so`の起動フックは通ります。
旧版`11b4b8e`は、QPU未指定の`init_post_opt()`と`task_init()`でエラーを
返すため、`required`設定ではスクリプト開始前に失敗します。

QRMIやプラグインのソース・バイナリを変更せずに試す場合は、
`spank_qrmi.so`の行の先頭を`required`から`optional`へ変更します。
このクラスタはconfigless構成なので、まず**vm1の**`/etc/slurm/plugstack.conf`
を更新し、vm2へ配信します。引数はそのまま維持します。

```text
optional /usr/lib64/slurm/spank_qrmi.so /etc/slurm/qrmi_config.json --env:RUST_LOG=qrmi=debug
```

vm1での変更・配信コマンドです。バックアップは既にある場合は上書きしません。

```bash
sudo cp -np /etc/slurm/plugstack.conf /etc/slurm/plugstack.conf.before-optional
sudo sed -i -E '/^[[:space:]]*required[[:space:]]+.*spank_qrmi[.]so/s/required/optional/' /etc/slurm/plugstack.conf
sudo cat /etc/slurm/plugstack.conf
sudo scontrol reconfigure
```

vm2では、配信された設定を次で確認します。

```bash
sudo cat /var/spool/slurm/d/conf-cache/plugstack.conf
```

`optional /usr/lib64/slurm/spank_qrmi.so ...`になっていれば反映されています。
configlessでは`/etc/slurm/plugstack.conf`がvm2になくても異常ではありません。
設定は`SlurmdSpoolDir/conf-cache/`に保存され、通常`/run/slurm/conf`からも
参照できます。[Slurmのconfigless仕様](https://slurm.schedmd.com/configless_slurm.html)
に従い、キャッシュは直接編集せずcontroller側を更新・再配信します。
このクラスタの`SlurmdSpoolDir`は`/var/spool/slurm/d`です。
キャッシュにもファイルがない場合や、再配信しても内容が変わらない場合は、
`scontrol show config`の`PlugStackConfig`と`SlurmdSpoolDir`、
vm2の`systemctl cat slurmd`で実際の設定参照先を確認します。
`qrmi_config.json`やプラグイン本体はこの設定配信とは別に各ノードへ配置します。

`optional`はプラグインを無効化する設定ではありません。
ロードして通常どおり実行し、その関数の失敗をジョブ起動の失敗にしない設定です。
[Slurm公式仕様](https://slurm.schedmd.com/spank.html#SECTION_CONFIGURATION)と、
[Slurm 22.05.11の実装](https://github.com/SchedMD/slurm/blob/slurm-22-05-11-1/src/common/plugstack.c#L756)
でも、optionalのエラーを正常戻り値へ変えることを確認しています。
[プラグインのREADME](https://github.com/qiskit-community/spank-plugins/blob/main/plugins/spank_qrmi/README.md)
にあるとおり、配信済みの設定は次のジョブ起動から読み込まれます。
このconfigless構成では上記の再配信を行い、クラスタ再起動は不要です。

この設定では、量子ジョブのQPU確保に失敗してもSlurmだけでは停止しなくなります。
今回のPoCは`task_runner`を呼ぶ前にBatchモードと取得トークンの存在を確認するため、
それらが欠けた場合はPython側で停止します。他の量子ジョブにも影響するクラスタ設定
なので、同等の検証があるとは限らない点を考慮してください。

vm2への配信も確認できたら、vm1がDRAIN状態なら復帰させます。
`idle*`なら登録待ちのため、少し時間を置いて再確認します。

```bash
sudo scontrol update NodeName=slurm-vm1 State=RESUME
sinfo -N -p debug -o '%N %T %E'
squeue
```

まず、QPUを使わない小さいジョブで起動を確認できます。

```bash
sbatch --wait --partition=debug --nodes=1 --ntasks=1 --output=hpc-smoke-%j.out --wrap='hostname'
```

既存のMPIジョブがまだPENDINGなら、その割り当てを待ちます。
既に失敗していれば、保存済み`input.bin`がある作業ディレクトリで、
実行ファイルの起動行を確認済みの`bitcount_retry.slurm`を再投入します。

```bash
sbatch --wait --output=retry.out --error=retry.err bitcount_retry.slurm
ls -l hist_u64.bin
```

生成されるヒストグラムは1024要素のuint64、8192バイトです。
手動再投入ではPrefectの失敗済みFlowは更新されません。

参考として`spank_qrmi_no_qpu.patch`も置いてありますが、optionalで回避する場合は
適用も再ビルドも不要です。ソース変更を伴わない恒久的な更新候補としては、
[upstreamの修正済みプラグイン](https://github.com/qiskit-community/spank-plugins/commit/a61aecf1f04a2936e5ebfa32c84bf0207364b66d)
への入れ替えがあります。この場合もSlurm 22.05とのビルド互換性と実機の確認は必要です。

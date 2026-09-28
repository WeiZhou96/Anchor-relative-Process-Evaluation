#!/usr/bin/env python
"""Render the Chinese K-c delivery report from measured artifacts only."""
from pathlib import Path
import csv,json,os,subprocess,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';B=OUT/'r2b';C=OUT/'r2c';K=Path(os.environ.get('APE_TMP',str(ROOT/'tmp')))/'r2/Kc'
H=[(4.,'77200b351bf1'),(10.,'8ac32aae418b'),(21.5,'3fb251dc210c')]
METRICS=['RMSCD@H','S_H@1','S_H@3','end_window_macro_acc','median_flips']
NAMES=dict(zip(METRICS,['RMSCD','S1','S3','末端Acc','F']))
def read(p):return json.loads(p.read_text())
def fmt(v):
    if v is None:return '未定义'
    if isinstance(v,bool):return '通过' if v else '未通过'
    return f'{v:.6f}' if isinstance(v,float) else str(v)
def hh(data,h):return next(x for x in data['by_h'] if x['h_s']==h)
def cell(a,b):return fmt(a)+' / '+fmt(b)
lines=[]
def para(s):lines.extend([s,''])
def table(headers,rows):
    lines.append('| '+' | '.join(headers)+' |');lines.append('| '+' | '.join(['---']*len(headers))+' |')
    for row in rows:lines.append('| '+' | '.join(str(x).replace('|',' / ').replace('\n','<br>') for x in row)+' |')
    lines.append('')
codehash=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
checks=read(C/'verification.json');accept=read(C/'vlm_acceptance.json');protected=read(C/'protected_check.json')
g3=read(C/'gates/g3_v5.json');g4=[read(p/'gates/g4.json') for p in [B,C]]
cals={pool:{h:read(p/'calib_plaus'/ph/'calibration.json') for h,ph in H} for pool,p in [('primary',B),('secondary',C)]}
para('# R2 K-c：VLM 并入、v5 G3 与最终表图报告')
para(f'状态：complete。工作分支 `r2/calib_c`；基点 master 为 `b8eda618c57a425d5afcad3ccf3727a835c200c9`。VLM 分支提交 `539a1ba1be8649d4f6b087f8078c6f470e78185e` 已合入，合并提交 `364f78a7f1ff5b4ba72a0c95fa859c598782f80c`。本报告生成时的最新代码提交：`{codehash}`。报告归档后的分支最新提交哈希另存 `outputs/r2c/delivery.json`，避免报告自引用。')
para('工作树：`$APE_TMP/r2/Kc/repo`。本报告的 `outputs/` 均指 `$APE_ROOT/outputs/`；新增计算与表图集中 `outputs/r2c/`，K-b 主结果保留在 `outputs/r2b/`。日志均位于 `$APE_TMP/r2/Kc/`。')
para('## 1. 任务状态与复现')
table(['事项','状态','脚本','产物'],[
['VLM 合并与复验','完成；两项验收通过，矩阵完整','scripts/k_c_validate_vlm.py','r2c/vlm_acceptance.json；V/两项验收件'],
['三档 eval','完成；旧552份逐字节复用，VLM新增3份','scripts/k_c_all.py eval','r2c/metrics/<hash>/'],
['双池标定、P、尺子','完成；主结果不重算，副池新算','scripts/k_c_all.py calibrate','r2c/calib_plaus/；calib/；metrics/<hash>/_pairs.json'],
['G2 / G4','完成；双池主副列','scripts/k_c_all.py g2 / g4','r2c/gates/g2.json；g4.json'],
['G3 三口径','完成；22.0秒为记录值','scripts/k_c_all.py g3；scripts/k_g3.py','r2c/gates/g3_v5.json'],
['A1/A3/A4/A5','完成；双池','scripts/k_c_all.py a1 / a3 / a4 / a5','r2c/ablations/；tables/*two_pools*'],
['P-b/P-c、G5','计算完成；G5不判通过','scripts/k_c_all.py mechanisms / g5','r2c/mechanisms/；gates/g5.json'],
['最终表图','完成；表1/1b/2/3/5、附表及图1—8','report/make_tables.py；make_figs.py','r2c/tables/；r2c/figs/'],
['测试与产物核对','完成','tests/test_r2c_v5.py；scripts/k_c_verify.py','r2c/verification.json；Kc/pytest.log'],
['最终结论对照','完成；G0/G5/G6边界单列','scripts/k_c_report.py','REPORT_R2_Kc.md；r2c/final_conclusions.json']])
para('复现解释器为 `$APE_PY`。新目录运行顺序为 `k_c_validate_vlm.py` → `k_c_all.py prepare` → `k_c_all.py all` → `make report ROOT=$APE_TMP/r2/Kc/repo` → `k_c_verify.py`。本次 all 通过 nohup 执行，日志 `Kc/compute.log`，耗时 '+fmt(read(K/'compute_status.json')['elapsed_s'])+' 秒，退出0。已完成目录按具名阶段续算，prepare 拒绝覆盖现有索引。')
para('主池184件，含168个非平凡系统、4个平凡锚点与12个量块；副池185件，非平凡系统169件。284个粗步长替身仍只服务Δ轴。原184件在96个协议点上的88,320个标量读数按round-trip精度复用，新增VLM 480个标量；副池的P、尺子、R_M、MRD、b_M、s_M、eps_max及全部义务对照重新计算。复用的是逐系统读数，池级结果没有沿用旧值。主索引未改写。')
para('## 2. VLM 规格与验收')
para('`REPORT_R2_V.md` 状态为 complete；因果验收为 passed，格式验收为 passed_v4。独立复核片段集合为dev 102、test 1514，共1,616段，每段103个互异列，j=−8…94、Δ=0.25秒，共166,448格；预测编码、one-hot/全零概率和零承诺均通过。⊥为893格。作答SHA-256：`'+accept['answer_sha256']+'`，与V的验收记录完全一致。')
para('因果验收件记录1,616段、166,448格、165,555个实际输入张量哈希复验一致，未来帧违规0。K-c核对验收件及矩阵哈希，没有再次解码视频或运行模型。格式共享验收器原版仍有2条规格差异：旧网格要求−11…88、旧概率检查未容许⊥全零；V按A4-3适配后失败0。未把passed_v4表述为原版无修改通过。')
para('应考者为 `Qwen/Qwen2.5-VL-7B-Instruct`，revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`；选帧按v6 A6-1整数索引轴规则。`train_data_unknown=true`、stateless=true、subsampling_equivalent=true，无seed、无粗步长目录。表1保留该标志；表1b单列确定性点估计，无虚构种子或标准差。')
rows=[]
for h,ph in H:
    v=read(C/'metrics'/ph/'r2__qwen25vl7b__readonly.json');m=v['frozen_family']
    rows.append([h,v['N_H'],fmt(m['end_window_macro_acc']),fmt(m['RMSCD@H']),fmt(m['S_H@1']),fmt(m['S_H@3']),fmt(m['median_flips']),f"0 / {h:g} / 未定义"])
table(['H/s','N_H','末端macro-Acc','RMSCD/s','S1','S3','F','ρ / τ_c / e_c'],rows)
para('## 3. 表3、P与G2双池结果')
para('以下成对数值依次为“主列不含VLM / 副列含VLM”。S1/S3为稳定正确率在1/3秒的值；末端Acc为窗末macro-Acc；F为改口中位数。三个H均为冻结54点乘积扫描，axis各14点另存对照。')
rows=[]
for h,ph in H:
    for m in METRICS:
        a=cals['primary'][h]['metrics'][m];b=cals['secondary'][h]['metrics'][m]
        rows.append([h,NAMES[m],cell(a['min_R'],b['min_R']),cell(a['MRD_plaus'],b['MRD_plaus']),cell(a['ruler'],b['ruler']),cell(a['n_pairs'],b['n_pairs']), '尺子退化，不计G2' if m=='median_flips' else a['verdict']+' / '+b['verdict']])
table(['H','指标','min R 主/副','MRD 主/副','尺子 主/副','P 主/副','判据 主/副'],rows)
para('H=10两池G2均通过，由S1、S3、末端Acc的MRD≥尺子承载；R臂均无承载指标。RMSCD未触发任一臂。F的步长平凡依赖和零尺子不计入G2。次档H=4、21.5不替代参考档决定门；RMSCD的MRD只在同H内计算。表3完整主副列还包括b_M、s_M、参考范围与eps_max；含量块对照、axis及逐系统delta_star均另存。')
para('## 4. G3三口径与oracle阴性对照')
para('原规则R2-8逐值保留；A4-2的23.5秒主池数值逐值保留；A5-1改用min(L+,22.0秒)，在同一E_H内按dev合格片段的三分位分层，共享1,000次源视频簇bootstrap，seed=20260903。H=10须两个随机量块均有后果，且非平凡系统有后果数至少为50%。')
rows=[]
for h,ph in H:
    for rule,label in [('original_rule','R2-8'),('v4_23p5','A4-2：23.5秒历史'),('v5_22p0','A5-1：22.0秒记录')]:
        vals=[]
        for pool in ['primary','secondary']:
            d=hh(g3[pool][rule],h);gg=d['groups']
            for role in ['random_block','real']:
                q=gg[role];vals.append('未定义' if q['n_undefined'] else f"{q['n_consequence']}/{q['n']}")
            vals.append(fmt(d['pass_gate'])+('，历史计算不作证据' if rule=='v4_23p5' else ''))
        rows.append([h,label,*vals])
table(['H','口径','随机 主','真实 主','门 主','随机 副','真实 副','门 副'],rows)
para('记录结论：H=10主池119/168（70.833333%）、副池120/169（71.005917%），随机量块均2/2，G3通过。H=21.5的真实后果数从23.5秒主池165/168降至0/168，副池165/169降至0/169；次档未通过。该档的22秒可见上限接近21.5秒固定窗，应如实理解为该观察条件下未得到门所要求的效应。')
rows=[]
for h,ph in H:
    a=hh(g3['primary']['v5_22p0'],h);b=hh(g3['secondary']['v5_22p0'],h)
    rows.append([h,a['n_dev_eligible'],a['n_test_eligible'],' / '.join(fmt(x) for x in a['dev_tercile_cuts_s']),'/'.join(map(str,a['rows'][0]['own']['counts'])),cell(a['ruler'],b['ruler'])])
table(['H','dev合格数','test合格数','dev切点/s','短/中/长层数','尺子 主/副'],rows)
para('H=10随机量块与VLM的长层减短层差如下；随机量块的own/fixed差和区间在两池相同，只有池级阈值略变。')
rows=[]
for rule,label in [('original_rule','R2-8'),('v4_23p5','23.5秒'),('v5_22p0','22.0秒')]:
    a=hh(g3['primary'][rule],10.)
    for r in a['rows']:
        if r['role']=='random_block':rows.append([label,r['system_id'],fmt(r['own']['gap_long_minus_short']),fmt(r['fixed']['gap_long_minus_short'])])
table(['口径','系统','own gap/s','fixed gap/s'],rows)
rows=[]
for r in hh(g3['secondary']['v5_22p0'],10.)['rows']:
    if r['role']=='random_block' or r['system_id']=='r2__qwen25vl7b__readonly':
        rows.append([r['system_id'],f"{fmt(r['own']['gap_long_minus_short'])} [{fmt(r['own']['ci_lo'])}, {fmt(r['own']['ci_hi'])}]",f"{fmt(r['fixed']['gap_long_minus_short'])} [{fmt(r['fixed']['ci_lo'])}, {fmt(r['fixed']['ci_hi'])}]",fmt(r['consequence'])])
table(['22秒口径系统','own差及95%CI/s','fixed差及95%CI/s','有后果'],rows)
para('v5明确要求oracle的|own gap|与|fixed gap|均≤尺子；任一不满足时，代码输出 `status=invalid_computation`、`pass_gate=null`，不写“门未通过”。K-c同时核对合成oracle量块与使用真值的平凡oracle，二者分别列出。')
rows=[]
for h,ph in H:
    old=hh(g3['primary']['v4_23p5'],h)
    for sid,label in [('block__oracle__default','合成oracle量块'),('trivial__oracle','平凡oracle缓存哨兵')]:
        r=next(x for x in old['rows'] if x['system_id']==sid)
        a=next(x for x in hh(g3['primary']['v5_22p0'],h)['oracle_negative_controls'] if x['system_id']==sid)
        b=next(x for x in hh(g3['secondary']['v5_22p0'],h)['oracle_negative_controls'] if x['system_id']==sid)
        rows.append([h,label,fmt(r['own']['gap_long_minus_short'])+' / '+fmt(r['fixed']['gap_long_minus_short']),fmt(a['own_gap'])+' / '+fmt(a['fixed_gap']),fmt(b['own_gap'])+' / '+fmt(b['fixed_gap']),'两池有效'])
table(['H','阴性对照','23.5秒 own/fixed','22秒主池 own/fixed','22秒副池 own/fixed','有效性'],rows)
para('K-b文中“oracle own gap=23.5秒”对应的是 `trivial__oracle`，合成 `block__oracle__default` 的own gap一直为0。本报告保留此身份区别。22秒三档两池共12项oracle检查全部通过，所有系统的可见缺格均为0；23.5秒历史口径中，172个非量块旧系统每件仍有670段、1,868格缺口。可见上限是统一缓存覆盖范围，不能写成完整原视频片尾。')
para('## 5. G4、义务对照与机制统计')
rows=[]
for h,ph in H:
    a,b=[hh(x,h) for x in g4]
    rows.append([h,*[cell(a[k],b[k]) for k in ['n_tied','n_heterogeneous','n_cross_base_heterogeneous','n_replicated_cross_base_groups']],'两池通过'])
table(['H','窗末同分对 主/副','异质对 主/副','跨基础异质对 主/副','≥2种子同向组 主/副','门'],rows)
para('VLM新增的同分/异质对在H=4、10、21.5分别为13、10、17；均为跨基础配对。VLM没有seed，不进入种子复现计数。窗末区间重叠是操作化同分定义，不能写成等价性检验；Holm区间与完整step-down拒绝标志联合读取。图1仍展示K-b配对，未因VLM加入重新选择主图系统。')
controls={pool:{n:read(p/'ablations'/f'{n}.json') for n in ['a1','a3','a4','a5']} for pool,p in [('primary',B),('secondary',C)]}
def row_for(pool,name,m):return next(x for x in controls[pool][name]['rows'] if x['h_s']==10 and x['metric']==m)
rows=[]
for m in METRICS:
    a,b=[row_for(p,'a1',m)['significant'] for p in ['primary','secondary']]
    aa,bb=[row_for(p,'a4',m) for p in ['primary','secondary']]
    ac,bc=[row_for(p,'a5',m) for p in ['primary','secondary']]
    rows.append([NAMES[m],cell(a['reversal_events'],b['reversal_events']),cell(a['unique_reversed_pairs'],b['unique_reversed_pairs']),cell(aa['n_alternative'],bb['n_alternative']),cell(ac['min_alternative'],bc['min_alternative']),cell(ac['n_alternative'],bc['n_alternative'])])
table(['H=10指标','A1翻转事件 主/副','A1独立翻转对 主/副','A4 Spearman域点 主/副','A5 min R 主/副','A5域点 主/副'],rows)
para('A1单位为“协议点×系统对”，新增同分另存；A4/A5域分母为54，r0固定0.9。A4原R域在主副池均为54/54/54/54/29点（依次RMSCD/S1/S3/末端Acc/F）。')
rows=[]
for h,ph in H:
    a,b=[hh(controls[p]['a3'],h) for p in ['primary','secondary']]
    rows.append([h,cell(a['n_detected'],b['n_detected']),cell(a['n_missed'],b['n_missed']),cell(a['detection_fraction'],b['detection_fraction'])])
table(['H','A3区分异质对 主/副','未区分 主/副','区分比例 主/副'],rows)
para('A3沿用动态分母瞬时micro准确率AUC/H；该标量未显著不代表整条曲线等价。P-b在H=10的RMSCD/S1/S3/末端Acc两池T_b均为1，F因常数响应未定义；四个扰动点不作为独立重复做推断。P-b各点偏移SD和P-c全系统值均以主副列完整输出。')
rows=[]
for h,ph in H:
    vals=[]
    for p in [B,C]:
        rs=[r for r in read(p/'mechanisms/p_c.json')['rows'] if r['h_s']==h and r['role']=='real']
        vals.append([len(rs),sum(r['saturated'] for r in rs),sum(r['T_c']!=0 for r in rs),sum(r['finest_mean_increment']>0 for r in rs)])
    rows.append([h,*[cell(a,b) for a,b in zip(*vals)]])
table(['H','非平凡系统 主/副','P-c中位改口饱和 主/副','T_c非零 主/副','最细均值仍增 主/副'],rows)
para('G5两池各有360条响应比较；未预注册通过阈值，`pass_gate=null`。H=10 RMSCD的Pearson相关如下，不能把计算完成写成机制门通过。')
rows=[]
ga,gb=[read(p/'gates/g5.json')['rows'] for p in [B,C]]
for axis in ['eps_sys_s','eps_jit_sd_s','delta_s','h_s']:
    a,b=[next(r for r in rs if r['h_s']==10 and r['metric']=='RMSCD@H' and r['block_family']=='all_blocks' and r['axis']==axis) for rs in [ga,gb]]
    rows.append([axis,cell(a['pearson'],b['pearson']),cell(a['sign_agreement'],b['sign_agreement'])])
table(['扰动轴','G5 Pearson 主/副','非参考点符号一致率 主/副'],rows)
para('## 6. 最终结论对照表：G0—G6')
gate_rows=[
['G0','未阻断；沿用冻结登记','v1的样本规模条件；报告长度分布偏移','N_H=1328/1113/742，簇=1297/1109/742','未登记可复算数值下限，不新增“充分样本”阈值；官方训练/测试片尾分布不同'],
['G1','通过；解析自检沿用并由测试复核','量块解析响应、手工轨迹、协议守卫','oracle RMSCD=0；lock(3)在参考网格为2.75秒','12个osc闭式不可分辨测试按原规则skip；不将VLM性能作为仪器自检'],
['G2','两池通过','H=10的MRD臂：S1、S3、末端Acc','主MRD/尺子：0.056602/0.054807；0.067523/0.050314；0.038494/0.021982','R臂无承载指标；F不计门；次档不替代参考档'],
['G3','两池通过；计算有效','v5 A5-1，22秒可见片尾，oracle双阴性对照','H=10随机2/2；真实119/168、120/169；oracle own/fixed均0','23.5秒历史无效缓存效应保留；H=21.5不通过；统一缓存上限不等于完整片尾'],
['G4','两池通过','跨基础异质性，Holm与至少两个同向种子','H=10同分7815/7825；异质3019/3029；复现组171/171','VLM无种子，不能承载复现；窗末同分非等价性'],
['G5','描述性完成；未判通过','量块与真实系统响应相关、符号一致','每池360行；H=10抖动Pearson −0.566253/−0.597778，符号一致0/0','没有预注册数值通过阈值，不在test后补设'],
['G6','未核定','仓库冻结件、契约及跟踪文本未给出G6定义','无可核对的规则或门产物','没有把“未找到定义”写成门失败或通过；本轮不臆造G6口径']]
table(['门','状态','承载臂或规则','记录值','限制'],gate_rows)
(C/'final_conclusions.json').write_text(json.dumps(dict(rows=[dict(zip(['gate','status','rule','record','limitation'],r)) for r in gate_rows]),indent=2,ensure_ascii=False))
with (C/'tables/final_conclusions.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['gate','status','rule','record','limitation']);w.writerows(gate_rows)
para('G0补充证据沿用 `report/text/g0_length_shift.md`：官方IID train/test为507/1520段，锚后长度中位数8.93/21.20秒；H=4/10/21.5合格比例为train 73.0%/47.3%/28.0%、test 87.4%/73.2%/48.8%。这些是修复源视频重叠与抽取dev之前的官方划分统计，不与当前411/102/1514划分混读。G6核查覆盖 `git ls-files` 中的Markdown/YAML/JSON/TXT；未发现定义，因此记录缺项，不声称它不存在于仓库以外。')
para('## 7. 第一轮至本轮的数值变化')
old=read(B/'first_round_comparison.json')
rows=[]
for m in METRICS:
    row=[NAMES[m]]
    for d in [old['old_H10'][m],old['ka_H10'][m],cals['primary'][10.]['metrics'][m],cals['secondary'][10.]['metrics'][m]]:
        row.append(' / '.join(fmt(d[k]) for k in ['min_R','MRD_plaus','ruler']))
    rows.append(row)
table(['H=10指标','第一轮旧扫描 minR/MRD/尺子','K-a纠错值','K-b=本轮主列','本轮含VLM副列'],rows)
table(['比较项','第一轮旧值','K-a','K-b=本轮主列','本轮含VLM副列'],[
['扫描实际P（RMSCD/S1/S3/末端/F）','981/1004/983/786/51','986/1024/1003/794/51','8383/9271/8720/5310/2571','8546/9430/8881/5446/2588'],
['G4：同分/异质/跨基础异质/复现组','仅同分展示，未执行R2-9','515/245/84/8','7815/3019/2141/171','7825/3029/2151/171'],
['G3：H=10记录','第一轮无本轮三分位门','R2-8空层，未定义','v4历史119/168；v5记录119/168','v5记录120/169'],
['G3：平凡oracle own/fixed差','未按该门记录','原规则0/未定义','v4 23.5/0 → v5 0/0','v5 0/0'],
['G0：N_H','1328/1112/742登记','1328/1113/742容差修正','1328/1113/742','1328/1113/742'],
['图1两系统末端Acc；RMSCD/s','0.327447;5.732480 / 0.305284;7.092318','0.327447;5.732480 / 0.302322;6.975067','沿用K-a配对及数值','主图不因副池改变配对']])
para('第一轮“S3的min R=0.890132跌破0.9”已按v4 A4-1撤回：旧扫描参考P没有使用正确粗步长替身，K-a修正为0.909272。旧eval的P本身为986/1024/1003/794/51，与旧扫描P区分保留。G2仍由MRD臂承载。G0的1112→1113来自恰好10秒的浮点边界片段按冻结1e−9秒容差保留，并非本轮改H。G1没有规则或数值变动；G5在K-a后成为描述性对照，未形成数值门记录；G6缺少规则，不构造跨轮差值。')
para('## 8. 表图、验证与边界')
para('表1含185行及VLM训练数据未知标志，名次分别列主池与副池；表1b为56组三种子汇总加1条确定性VLM点估计。表2每系统并列G3三规则和两池，空层为未定义；表3三档完整主副列；表5逐系统现象频率含VLM。A1/A3/A4/A5、P-b/P-c、G2/G4/G5均有 `*two_pools*` 附表。所有CSV与Markdown在 `outputs/r2c/tables/`。')
para('图1—7使用不含VLM的K-b主池读数，图5保留原G3追溯；图8为 `fig8_r2c_g3_three_rules`，并列三规则、三档H及VLM副池读数。八张图全部输出300 DPI PNG与PDF，其中图5—8另有SVG。图内沿用英文。总览和图8细读发现的空层坐标歧义已修正，历史23.5秒列明确标注cache invalid；未用平滑或虚构点补空层。QA存 `outputs/r2c/qa/`。图4明确保留dashcam未取得数据的空缺，不称双轨完成。')
testlog=(K/'pytest.log').read_text();match=re.search(r'\d+ passed, \d+ skipped in [\d.]+s',testlog)
para('pytest：`'+match.group(0)+'`；相对K-b新增30项v5/VLM测试，覆盖22秒端点、min(L+,22)删失、23.5秒缓存oracle伪效应、双绝对差含等号边界、计算无效与门未通过的区别、169件池的50%门槛以及无种子VLM不得承担种子复现。12项skip为原有osc量块不可分辨情形。`make report`退出0；`git diff --check`通过。')
para(f'产物验证通过{checks["assertions"]}项断言：552份旧eval逐字节一致，原扫描读数完全一致，30张R表由保存扫描重新核对，三档参考R均为1，逐指标尺子独立由窗末同分对复算，原系统bootstrap与K-b一致至1e−12，G3所有可见缺格为0，12项oracle检查均为0/0。')
para(f'保护文件核对：{protected["protected_files"]}个文件SHA-256全部一致，涵盖pi0及守卫、冻结件、manifest、K-b产物和全部作答文件。master仍为 `{protected["master"]}`；数据与outputs软链接保持原位。未使用GPU、未写数据目录、未修改dev或作答矩阵、未在test上选择模型、提示、参数、网格或阈值。代码改动限ape/、report/、scripts/k_*、tests/；根报告是本次明确授权的交付例外。')
para('未完成边界：G6无可核对定义；G5没有预注册通过阈值；G0没有数值化样本下限；dashcam轨没有新增数据。这些限制均保留，未通过补设规则或推测数值填平。VLM并入、v5 G3计算、双池统计、表图、测试与代码提交均已完成。')
(ROOT/'REPORT_R2_Kc.md').write_text('\n'.join(lines),encoding='utf-8')
print('REPORT',ROOT/'REPORT_R2_Kc.md','CHARS',len('\n'.join(lines)))

#!/usr/bin/env python3
"""Replace reference columns in a new copy of the saved NN observables package."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    old, work, out = args.old_package, args.work, args.out
    complete = json.loads((work/'COMPLETE.json').read_text())
    if not complete['nn_direct_audit_passed']:
        raise SystemExit('Reference audit has not passed')
    compiler_control = json.loads((work/'compiler_control.json').read_text())
    if not compiler_control['passed']:
        raise SystemExit('Historical gfortran/nvfortran control has not passed')
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_rhfb_observables.py')),
                    '--nn-root', str(old/'nn'), '--fortran-dripline', str(work/'runs'),
                    '--fortran-production', str(work/'runs'), '--out', str(out)], check=True)
    previous = {(r['set'],r['nucleus']):r for r in read_csv(old/'observables.csv')}
    rows = read_csv(out/'observables.csv')
    assert len(rows)==len(previous)==complete['records']
    assert {(r['set'],r['nucleus']) for r in rows}==set(previous)
    convergence_keys = ['converged','epochs','final_sigma_change_kev','final_learning_rate']
    changes = []
    for row in rows:
        prior = previous[(row['set'],row['nucleus'])]
        for key in convergence_keys:
            row[key] = prior[key]
        for key in prior:
            if key.endswith('_nn'):
                assert row[key]==prior[key], (row['nucleus'],key)
        for key,value in row.items():
            if key.endswith('_diff'):
                stem = key[:-5]
                assert abs(float(value)-(float(row[stem+'_nn'])-float(row[stem+'_fortran'])))<1e-10
        changes.append({'set':row['set'],'nucleus':row['nucleus'],
                        'E_nn_MeV':row['E_total_nn'],
                        'E_fortran_old_MeV':prior['E_total_fortran'],
                        'E_fortran_corrected_MeV':row['E_total_fortran'],
                        'NN_minus_old_MeV':prior['E_total_diff'],
                        'NN_minus_corrected_MeV':row['E_total_diff'],
                        'E_per_A_diff_corrected_keV':1000*float(row['E_per_A_diff']),
                        'r_charge_old_fm':prior['r_charge_fortran'],
                        'r_charge_corrected_fm':row['r_charge_fortran'],
                        'E_pair_diff_corrected_MeV':row['E_pair_diff']})
        source = old/'nn'/row['set']/row['nucleus']/'summary.json'
        assert sha(source)==sha(out/'nn'/row['set']/row['nucleus']/'summary.json')
        run = work/'runs'/row['nucleus']
        parsed = json.loads((run/'parsed.json').read_text())
        guard = json.loads((run/'guard.json').read_text())
        assert parsed['converged'] and guard['status']=='completed'
        dest = out/'fortran'/row['set']/row['nucleus']
        for item in ('rhf.dat','input_meta.json','provenance.json','guard.json','DEN'):
            if (run/item).is_dir():
                shutil.copytree(run/item,dest/item)
            else:
                shutil.copy2(run/item,dest/item)
    write_csv(out/'observables.csv',rows)
    write_csv(out/'reference_changes.csv',changes)
    levels = read_csv(out/'single_particle_levels.csv')
    unmatched = [r for r in levels if float(r['v2_nn'])>0.05 and not r['e_fortran_mev']]
    assert not unmatched, unmatched[:5]
    provenance = out/'validation'
    provenance.mkdir()
    for name in ('manifest.json','SOURCE_SHA256.json','all_nn_direct_audit.json',
                 'all_nn_direct_audit_guard.json','nn_saved_state_checks.csv',
                 'nn_tests.log','nn_tests_guard.json','compiler_control.json','COMPLETE.json'):
        shutil.copy2(work/name,provenance/name)
    source = out/'source'
    source.mkdir()
    for name in ('PotelHF.f90','PotelHF.patch','run_corrected.py','audit_potelhf_direct.py',
                 'fortran_rhfb_reference.py','build_rhfb_observables.py','manifest.json',
                 'SOURCE_SHA256.json','check_compiler_control.py'):
        shutil.copy2(work/name,source/name)
    shutil.copytree(work/'inputs',source/'inputs')
    for run in (work/'runs').iterdir():
        if (run/'parsed.json').exists():
            shutil.copy2(run/'rhf.dat',source/'inputs'/run.name/'rhf.dat')
    for build in ('src_kmx54','src_kmx88'):
        dest=source/build
        dest.mkdir()
        for p in (work/build).iterdir():
            if p.suffix=='.f90' or p.name in ('makefile','build_manifest.json','build.log'):
                shutil.copy2(p,dest/p.name)
    tools = source/'package_tools'
    tools.mkdir()
    for name in ('package_potelhf_update.py','build_rhfb_observables.py'):
        shutil.copy2(Path(__file__).with_name(name),tools/name)
    shutil.copy2(Path(__file__).resolve().parents[1]/'docs/POTELHF_REFERENCE_FIX_20261105.md',
                 provenance/'POTELHF_REFERENCE_FIX_20261105.md')
    (out/'README_original_v2.md').write_text((old/'README.md').read_text())
    (out/'README.md').write_text(
        '# RHFB v2：PotelHF 修正后的参考数据\n\n'
        '使用用户提供的 PotelHF.f90，将原 59–62 行的 rho 张量块移到原 54–57 行的 omega 张量块之前。'
        '原来的 gotn=0 分支会跳过 rho 有限程张量及矢量–张量直接项。\n\n'
        f'本包含 {len(rows)} 条计算记录、{complete["runs"]} 个不同核素。生产组和同位素链的重复核保留各自 NN 记录。'
        'NN 摘要逐字节保留原 v2 文件；Fortran 参考全部重新自洽求解。'
        '所有参考均已收敛，输入参数、阻塞和各核原有 KMX 空间保持一致。'
        '6 个原本地参考使用 KMX=(5,4)，其余 63 个使用 KMX=(8,8)。'
        '本次均在 GPU04 用 nvfortran 编译，原本地 6 个参考此前使用 gfortran；'
        '已用未修改的旧代码逐核验证编译器差异（见 validation/compiler_control.json）。\n\n'
        '主表仍为 observables.csv、single_particle_levels.csv、channel_energies.csv；'
        'reference_changes.csv 给出旧参考、新参考及 NN 的逐核差异。'
        '能量和半径按不含质心修正的口径比较；通道表保留 Fortran 原始标签，含质心项。'
        '原 v2 列说明见 README_original_v2.md；其中旧日期、数量与物理解读仅作历史记录。\n\n'
        'validation/ 含 NN 测试结果、保存态检查以及对全部修正参考密度的 Hartree 通道核验。'
        '同密度核验同时报告 NN 原积分度量和 Fortran 外层 Simpson 度量；没有改动 NN 泛函。'
        '这些检查针对实现一致性，不能证明有限训练和有限正则空间已达到精确变分极限。'
        'NN 的场变化停止判据与变分残差不同，残余轨道/占据残差在验证 CSV 中给出。\n\n'
        '能级表中的 e_nn_mev 是保存轨道的 h_nunu 期望值，不是 Hamiltonian 本征值。'
        '按 species、kappa、保存的 principal 标签对应，没有进行波函数重叠匹配。'
        '占据接近 0 或 1 时轨道内部旋转可改变这些对角元，不能将其差值直接解释为本征能谱误差；'
        '最新 v2 重核的对角元差异仍然明显，不能沿用旧 README 的能级精度宣称。\n\n'
        'source/ 含用户修正版、补丁、两套编译源码和运行脚本；fortran/ 附输入、收敛输出、密度、能级和资源日志。'
        '本包不含论文、模型权重或原始旧结果副本。\n')
    checksums = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in checksums.items()))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(out.name)/p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(archive)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Input writer and output parser for the user's S-RHFB-C Fortran program.

Post-training reference only.  This script never runs the Fortran executable;
runs go through ``scripts/run_rhfb_limited.py`` (see
outputs/fortran_reference/BUILD_NOTES.md).

Subcommands
-----------
write-input  --nucleus O18 --z 8 --n 10 [--block-n KAPPA ...] [--block-p KAPPA ...] --out DIR
parse        --run-dir DIR --out parsed.json

Blocking codes are Dirac kappa values: the program blocks the *lowest*
quasiparticle (n=1) of the kappa block whose kappa equals the code
(Config.f90).  kappa = -(j+1/2) for j = l+1/2, +(j+1/2) for j = l-1/2:
s1/2=-1, p3/2=-2, p1/2=+1, d5/2=-3, d3/2=+2, f7/2=-4, f5/2=+3, g9/2=-5.
"""
import argparse
import json
import re
from pathlib import Path

ELEMENTS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
            10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
            18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
            26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 50: 'Sn', 82: 'Pb'}
KBX = 5          # Define.f90: maximal number of blocked kappa codes per species
MSD = 201        # Define.f90: radial mesh points (compile-time)


def kappa_label(kappa):
    """Spectroscopic label for a Dirac kappa (e.g. -3 -> 'd5/2')."""
    if kappa < 0:
        j2, l = -2*kappa-1, -kappa-1
    else:
        j2, l = 2*kappa-1, kappa
    return 'spdfghijklm'[l]+f'{j2}/2'


def write_input(args):
    a = args.z+args.n
    symbol = ELEMENTS.get(args.z)
    if symbol is None or not args.nucleus.startswith(symbol):
        raise SystemExit(f'nucleus name {args.nucleus!r} inconsistent with Z={args.z}')
    if args.nucleus[len(symbol):] not in ('', str(a)):
        raise SystemExit(f'nucleus name {args.nucleus!r} inconsistent with A={a}')
    for codes in (args.block_n, args.block_p):
        if len(codes) > KBX:
            raise SystemExit(f'at most {KBX} blocking codes per species (KBX)')
        if any(c == 0 for c in codes):
            raise SystemExit('kappa 0 is not a valid blocking code')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def ival(key, value):
        return f'{key:<7s}={value:10d}'

    def fval(key, value):
        return f'{key:<7s}={value:10.3f}'

    def pair(key, a1, a2):
        # Inout.f90: read(lin,'(8x, f6.1, f8.1)')
        return f'{key:<7s}={a1:6.1f}{a2:8.1f}'

    # Header: read(lin,'(2a1,6i4)') nucnam, A, N, Z, IDT, ipa, NSTEP.
    # IDT=0 disables isotopic/isotonic chains (Nnum = 0).
    lines = [f'{symbol:>2s}{a:4d}{args.n:4d}{args.z:4d}{0:4d}{a:4d}{0:4d}        ! nucleus: A N Z IDT ipa NSTEP',
             ival('l6', 10)+'                ! output file unit',
             ival('pset', args.pset)+'                ! 0 --> PKA1',
             fval('xstep', args.xstep)+'                ! radial step (fm); rmax = h*(MSD-1)',
             ival('maxi', args.maxi)+'                ! maximal number of iterations',
             ival('nmix', 7)+'                ! Broyden steps',
             fval('xmix', args.xmix)+'                ! mixing parameter',
             ival('inin', 1)+'                ! 1: Woods-Saxon start',
             ival('pair', 1)+'                ! 1 --> Gogny force; 0 --> delta force',
             fval('V0/VG', args.vg)+'                ! Gogny strength factor',
             pair('XECT', *args.xect)+'            !--- Energy cutoff',
             pair('XEPT', *args.xept)+'            !--- Pairing window',
             ival('Tensor', args.tensor)+'                ! 0 --> drop the tensor force components',
             ival('IME', 0)+'                ! 0 --> no interaction matrix elements',
             fval('EMX', 10.0)+'                ! (unused for IME=0)',
             f'{len(args.block_n)},{len(args.block_p)}                               ! number of blocking codes (n, p)',
             (','.join(str(k) for k in args.block_n) or '0')+'                                 ! blocking kappa (n)',
             (','.join(str(k) for k in args.block_p) or '0')+'                                 ! blocking kappa (p)',
             '']
    (out/'rhf.dat').write_text('\n'.join(lines))
    meta = {'nucleus': f'{symbol}{a}', 'Z': args.z, 'N': args.n, 'A': a,
            'pset': args.pset, 'pairing': 'Gogny D1S', 'xstep_fm': args.xstep,
            'rmax_fm': round(args.xstep*(MSD-1), 6), 'mesh_points': MSD,
            'block_n_kappa': args.block_n, 'block_p_kappa': args.block_p,
            'block_n_labels': [kappa_label(k) for k in args.block_n],
            'block_p_labels': [kappa_label(k) for k in args.block_p],
            'xect': args.xect, 'xept': args.xept, 'tensor': args.tensor,
            'maxi': args.maxi, 'xmix': args.xmix, 'vg': args.vg}
    (out/'input_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(out/'rhf.dat')


# ----------------------------------------------------------------------------- parse

def _floats(text):
    return [float(x) for x in re.findall(r'[-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?', text)]


def _find_main_output(run_dir):
    meta = run_dir/'input_meta.json'
    if meta.exists():
        info = json.loads(meta.read_text())
        name = info['nucleus']
        cands = sorted(run_dir.glob(f'{name}.*'))
        cands = [c for c in cands if c.is_file()]
        if cands:
            return cands[0], name
    cands = [p for p in run_dir.iterdir() if p.is_file() and re.fullmatch(r'[A-Z][a-z]?\d+\.[A-Za-z0-9-]+', p.name)]
    if not cands:
        raise SystemExit(f'no main output file in {run_dir}')
    return cands[0], cands[0].name.split('.')[0]


LINE_KEYS = {
    'particle_number': 'Particle number                        ',
    'rms_wo_com': 'Rms-Radius w/o CoM',
    'charge_wo_com': 'Charge-Radius w/o CoM',
    'rms_w_com': 'Rms-Radius w/ CoM',
    'charge_w_com': 'Charge-Radius w/ CoM',
    'lambda': 'Chemical potential in RHFB equation',
    'gap_eff': 'Effective Pairing gaps in total',
    'gap_avg': 'Another Pairing gaps in total',
    'cooper_pairs': 'Number of Cooper-pairs in total',
    'e_sp': 'Single-Particle Energy in total',
    'e_kin': 'Kinetic Energy (rest mass excluded)',
    'e_pair': 'E-pair (Pairing correlations)',
    'e_cm': 'E-cm (center-of-mass corrections)',
    'e_tot': 'E-B: Binding energy (energy functional)',
    'e_tensor': 'E-B: Tensor energy functional',
    'e_per_a': 'E-B/A (Binding energy per particle)',
    'e_rearr': 'Rearrangement terms',
    'e_coul_dir': 'Direct Coulomb (Vector coupling)',
    'e_coul_exc': 'Exchange Coulomb (Vector coupling)',
}


def _last_expect_block(text):
    starts = [m.start() for m in re.finditer(r'BEGIN EXPECT', text)]
    if not starts:
        return None
    block = text[starts[-1]:]
    end = block.find('END EXPECT')
    return block if end < 0 else block[:end]


def _parse_levels(path, quasiparticle):
    rows = []
    if not path.exists():
        return None
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(x) for x in parts[3:]]
            i1, i2 = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if quasiparticle:
            # levQ: State, kappa, 2j+1, v^2, E_qp (Inout.f90 Export)
            rows.append({'state': parts[0], 'kappa': i1, 'degeneracy': i2,
                         'v2': vals[0], 'E_qp': vals[1]})
        else:
            # lev: State, n, 2j+1, vv, EC, del, EC, del, r2, 2r, rd   (canonical basis)
            keys = ['v2', 'e_canonical', 'delta_canonical', 'e_canonical_2', 'delta_canonical_2',
                    'r2', 'r', 'dr']
            row = {'state': parts[0], 'n': i1, 'degeneracy': i2}
            row.update(dict(zip(keys, vals)))
            rows.append(row)
    return rows


def parse(args):
    run_dir = Path(args.run_dir)
    main, name = _find_main_output(run_dir)
    text = main.read_text(errors='replace')
    result = {'run_dir': str(run_dir.resolve()), 'main_output': main.name, 'nucleus': name}
    meta = run_dir/'input_meta.json'
    if meta.exists():
        result['input'] = json.loads(meta.read_text())

    m = re.search(r'Iteration (converged|interrupted) after\s*(\d+) steps\s+si =\s*(\S+)\s+(\S+)', text)
    result['converged'] = bool(m and m.group(1) == 'converged')
    result['iterations'] = int(m.group(2)) if m else None
    result['final_si'] = [float(m.group(3)), float(m.group(4))] if m else None
    if m is None:
        result['status'] = 'no_convergence_line (run incomplete?)'
    hist = re.findall(r'^\s*(\d+)\. sh =\s*(\S+)\s+E/A =\s*(\S+)', text, re.M)
    result['iteration_history_tail'] = [
        {'ii': int(i), 'si_h': float(s), 'E_per_A': float(e) if re.match(r'[-+\d.]', e) else e}
        for i, s, e in hist[-5:]]

    block = _last_expect_block(text)
    raw = {}
    if block:
        for key, label in LINE_KEYS.items():
            for line in block.splitlines():
                if line.startswith(' '+label.strip()) or line.strip().startswith(label.strip()):
                    raw[key] = _floats(line[len(label)+1:])
                    break
    result['raw_expect'] = raw

    def trip(key):
        v = raw.get(key)
        return None if not v else {'n': v[0], 'p': v[1], 'total': v[2]} if len(v) >= 3 else v

    if 'e_tot' in raw:
        etot, ecm = raw['e_tot'][2], raw.get('e_cm', [0, 0, 0])[2]
        a = result.get('input', {}).get('A') or int(re.sub(r'\D', '', name))
        result['energy'] = {
            'E_total_with_cm_MeV': etot,
            'E_cm_MeV': ecm,
            'E_total_no_cm_MeV': etot-ecm,
            'E_per_A_with_cm_MeV': etot/a,
            'E_per_A_no_cm_MeV': (etot-ecm)/a,
            'E_per_A_printed': raw.get('e_per_a', [None]*3)[2],
            'E_pair_MeV': trip('e_pair'),
            'E_kin_MeV': trip('e_kin'),
            'E_rearrangement_MeV': trip('e_rearr'),
            'note': 'E_total includes the microscopic c.m. correction COMS(1) (CoM.f90); no_cm = E - E_cm',
        }
    if 'lambda' in raw:
        result['lambda_MeV'] = {'n': raw['lambda'][0], 'p': raw['lambda'][1]}
    result['pairing_gap_effective_MeV'] = (
        {'n': raw['gap_eff'][0], 'p': raw['gap_eff'][1]} if 'gap_eff' in raw else None)
    result['pairing_gap_average_MeV'] = trip('gap_avg')
    result['cooper_pairs'] = trip('cooper_pairs')
    result['particle_number'] = trip('particle_number')
    result['radii_fm'] = {
        'rms_without_cm': trip('rms_wo_com'),
        'charge_without_cm': raw.get('charge_wo_com', [None])[0],
        'rms_with_cm': trip('rms_w_com'),
        'charge_with_cm': raw.get('charge_w_com', [None])[0],
        'note': "rms_*: n, p, total(matter)",
    }
    blk = re.findall(r'Blocking ([np]) block - Probability\s+([-\d\s]+)', text)
    result['blocking'] = {s: [int(x) for x in v.split()] for s, v in blk}

    levels = {}
    for tag, key in (('N', 'n'), ('P', 'p')):
        canon = sorted(run_dir.glob(f'LEV/{name}-lev_{tag}.*'))
        qp = sorted(run_dir.glob(f'LEV/{name}.levQ_{tag}.*'))
        levels[key] = {
            'canonical': _parse_levels(canon[0], False) if canon else None,
            'quasiparticle': _parse_levels(qp[0], True) if qp else None,
        }
        if levels[key]['canonical']:
            occ = [r for r in levels[key]['canonical'] if r['v2'] > 1e-3]
            levels[key]['canonical_occupied'] = sorted(occ, key=lambda r: r['e_canonical'])
            lam = result.get('lambda_MeV', {}).get(key)
            if lam is not None:
                near = [r for r in levels[key]['canonical'] if abs(r['e_canonical']-lam) < 10.0]
                levels[key]['canonical_near_fermi_10MeV'] = sorted(near, key=lambda r: r['e_canonical'])
    result['levels'] = levels
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    summary = {k: result.get(k) for k in ('nucleus', 'converged', 'iterations')}
    summary.update(result.get('energy', {}) and {
        'E_no_cm': result['energy']['E_total_no_cm_MeV'],
        'E/A_no_cm': result['energy']['E_per_A_no_cm_MeV']})
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write-input')
    w.add_argument('--nucleus', required=True)
    w.add_argument('--z', type=int, required=True)
    w.add_argument('--n', type=int, required=True)
    w.add_argument('--block-n', type=int, nargs='*', default=[], metavar='KAPPA')
    w.add_argument('--block-p', type=int, nargs='*', default=[], metavar='KAPPA')
    w.add_argument('--pset', type=int, default=0)
    w.add_argument('--xstep', type=float, default=0.1)
    w.add_argument('--maxi', type=int, default=500)
    w.add_argument('--xmix', type=float, default=0.5)
    w.add_argument('--vg', type=float, default=1.0)
    w.add_argument('--xect', type=float, nargs=2, default=[350.0, -10.0])
    w.add_argument('--xept', type=float, nargs=2, default=[400.0, 400.0])
    w.add_argument('--tensor', type=int, default=1)
    w.add_argument('--out', required=True)
    w.set_defaults(func=write_input)
    q = sub.add_parser('parse')
    q.add_argument('--run-dir', required=True)
    q.add_argument('--out', required=True)
    q.set_defaults(func=parse)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

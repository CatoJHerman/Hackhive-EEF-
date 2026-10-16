"""Read-only source audit and separate candidate exports. Never trains or splits data.

Requires numpy, pandas and openpyxl. Example:
python audit_datasets.py --obd /path/to/MASTER_TRAIN_X.xlsx
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_FEATURES = ['rms', 'peak_amplitude', 'crest_factor', 'dominant_frequency_hz']
AUDIO_COLUMNS = ['recording_id', 'start_seconds', 'end_seconds', 'sample_rate', *AUDIO_FEATURES]
OBD_FEATURES = ['LOAD_PCT', 'ECT', 'MAP', 'RPM', 'VSS', 'IAT', 'MAF', 'FRP', 'BARO', 'VPWR', 'AAT']
AUDIO_SOURCE = 'https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(audio_path, obd_path, output):
    output.mkdir(parents=True, exist_ok=True)
    audio = pd.read_csv(audio_path, encoding='utf-8-sig')
    if list(audio.columns) != AUDIO_COLUMNS:
        raise ValueError('Audio schema differs from the shared eight-column export.')
    numeric = audio[AUDIO_COLUMNS[1:]].apply(pd.to_numeric, errors='raise')
    if audio.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Audio contains missing or non-finite values.')
    if audio.recording_id.astype(str).str.strip().eq('').any():
        raise ValueError('Blank recording ID.')
    audio[AUDIO_COLUMNS[1:]] = numeric
    checks = {
        'positive_sample_rate': bool((audio.sample_rate > 0).all()),
        'one_second_windows': bool(np.allclose(audio.end_seconds-audio.start_seconds, 1)),
        'valid_times': bool(((audio.start_seconds >= 0) & (audio.end_seconds > audio.start_seconds)).all()),
        'nonnegative_features': bool((audio[AUDIO_FEATURES] >= 0).all().all()),
        'peak_at_least_rms': bool((audio.peak_amplitude + 1e-8 >= audio.rms).all()),
        'frequency_below_nyquist': bool((audio.dominant_frequency_hz <= audio.sample_rate/2).all()),
        'crest_matches_peak_over_rms': bool(np.allclose(audio.crest_factor,
            np.divide(audio.peak_amplitude, audio.rms, out=np.zeros(len(audio)), where=audio.rms.to_numpy()!=0), rtol=1e-5, atol=1e-6)),
        'unique_window_keys': not bool(audio.duplicated(['recording_id','start_seconds','end_seconds']).any()),
    }
    if not all(checks.values()):
        raise ValueError(f'Audio checks failed: {checks}')
    audio.to_csv(output/'audio_validated.csv', index=False)
    manifest = pd.DataFrame({'recording_id':audio.recording_id.unique()})
    for col in ['parent_recording_id','vehicle_id','session_id','source_url','label','label_basis','permission_to_share']:
        manifest[col] = 'unknown'
    manifest['reported_folder'] = 'normal_engine_idle'
    manifest['source_url'] = AUDIO_SOURCE
    # Do not overwrite metadata that a user has already filled in.
    manifest_path = output/'audio_metadata_to_complete.csv'
    if not manifest_path.exists():
        manifest.to_csv(manifest_path, index=False)

    obd = pd.read_excel(obd_path, sheet_name='Train')
    codes = [c for c in obd if c.startswith('P') and c[1:].isdigit()]
    if not set(OBD_FEATURES+['P0000','Mode']).issubset(obd.columns) or len(codes)!=14:
        raise ValueError('Unexpected OBD schema.')
    if obd.isna().any().any() or not np.isfinite(obd.to_numpy(dtype=float)).all():
        raise ValueError('OBD contains missing or non-finite values.')
    if not obd[codes].isin([0,1]).all().all():
        raise ValueError('OBD code columns are not binary.')
    fault_codes = [c for c in codes if c != 'P0000']
    any_code = obd[fault_codes].any(axis=1)
    complement_ok = bool((obd.P0000 == (~any_code).astype(int)).all())
    unique = obd.drop_duplicates().copy()
    unique.insert(0, 'source_excel_row', unique.index+2)
    unique['recorded_fault_code_present'] = unique[fault_codes].any(axis=1).astype(int)
    ambiguous = unique.P0000.eq(unique['recorded_fault_code_present'])
    unique['label_status'] = np.where(ambiguous, 'ambiguous', 'explicit_code_flags')
    unique['recorded_fault_code_present'] = unique['recorded_fault_code_present'].astype('Int64')
    unique.loc[ambiguous, 'recorded_fault_code_present'] = pd.NA
    unique.to_csv(output/'obd_unique_candidates.csv', index=False)
    unique.loc[ambiguous].to_csv(output/'obd_labels_to_review.csv', index=False)
    unique.loc[~ambiguous].to_csv(output/'obd_explicit_labels.csv', index=False)
    membership = obd.copy()
    membership.insert(0, 'source_excel_row', membership.index+2)
    membership['exact_row_group'] = pd.util.hash_pandas_object(obd, index=False).astype(str)
    membership[['source_excel_row','exact_row_group']].to_csv(output/'obd_duplicate_membership.csv', index=False)
    # Identical sensor inputs with different labels are worth investigating, not deleting silently.
    conflicting = obd.groupby(OBD_FEATURES, dropna=False)[codes].nunique().gt(1).any(axis=1)
    identical_codes = [[a,b] for i,a in enumerate(codes) for b in codes[i+1:] if obd[a].equals(obd[b])]
    contract = {
        'audio': {'features':AUDIO_FEATURES, 'sample_rate_hz':44100, 'window_seconds':1,
                  'source_url_as_supplied_by_user':AUDIO_SOURCE,
                  'labels_verified':False, 'split_created':False,
                  'exclude_from_inputs':['recording_id','start_seconds','end_seconds','sample_rate','label','source_url','parent_recording_id'],
                  'blockers':['Source/permission and labels unknown','Parent recording/session groups unknown','No labelled abnormal examples supplied']},
        'obd': {'features':OBD_FEATURES, 'binary_target':'recorded_fault_code_present',
                'target_definition':'1: other code present and P0000=0. 0: P0000=1 and all other codes zero. All-zero or conflicting flags are quarantined with a blank target. Predicts code-label status, not future failure.',
                'exclude_from_inputs':codes+['Mode','source_excel_row','recorded_fault_code_present','label_status'],
                'units':'Original values retained; source-unit discrepancy unresolved; no conversions applied.',
                'split_created':False,
                'blockers':['No vehicle/session IDs or timestamps in supplied workbook','No independent test workbook supplied','Confirm units and Mode semantics'],
                'source':'https://github.com/AbouAbdallah-Lounis/OBD-Dataset',
                'license_as_documented':'CC BY 4.0; cite Karim Lounis, Hadjer Mekki, and Wissam Hadjab, DieselOBD (2025).'},
        'rules':['Do not join unrelated audio and OBD rows.',
                 'Fit scaling and imputation on training data only.',
                 'Split by verified parent recording/session/vehicle before fitting.',
                 'Deduplication does not make rows independent; no random row split for final claims.',
                 'Do not infer vehicle groups from fault labels.',
                 'Do not treat CSV row order as a measured time axis.'],
    }
    summary = {
        'source_sha256':{'audio':digest(audio_path),'obd':digest(obd_path)},
        'audio':{'rows':len(audio),'recording_ids':int(audio.recording_id.nunique()),
                 'duplicates':int(audio.duplicated().sum()), 'duplicate_feature_rows':int(audio.duplicated(AUDIO_FEATURES).sum()),
                 'checks':checks,'ranges':audio[AUDIO_FEATURES].agg(['min','max']).to_dict()},
        'obd':{'rows':len(obd),'columns':len(obd.columns),'duplicates':int(obd.duplicated().sum()),
               'unique_rows':len(unique),'missing_cells':int(obd.isna().sum().sum()),
               'p0000_complements_other_codes':complement_ok,
               'raw_binary_counts':any_code.astype(int).value_counts().to_dict(),
               'unique_binary_counts':{str(k):int(v) for k,v in unique.recorded_fault_code_present.value_counts().items()},
               'ambiguous_raw_rows':int(obd.P0000.eq(any_code.astype(int)).sum()),
               'ambiguous_unique_rows':int(ambiguous.sum()),
               'explicit_label_candidate_rows':int((~ambiguous).sum()),
               'unique_label_patterns':len(obd[codes].drop_duplicates()),
               'conflicting_sensor_groups':int(conflicting.sum()),'identical_code_columns':identical_codes,
               'ranges':obd[OBD_FEATURES].agg(['min','max']).to_dict()},
    }
    for name,data in [('audit_summary.json',summary),('modelling_contract.json',contract)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    # Verify round-trip exports and source immutability.
    assert len(pd.read_csv(output/'audio_validated.csv')) == len(audio)
    assert len(pd.read_csv(output/'obd_unique_candidates.csv')) == len(unique)
    assert summary['source_sha256']=={'audio':digest(audio_path),'obd':digest(obd_path)}
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',type=Path,default=Path('baselines/normal_idle_features_1s.csv'))
    parser.add_argument('--obd',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/pre_modelling'))
    args=parser.parse_args()
    run(args.audio,args.obd,args.output)

"""CSC-4240 offline audio forensics: train models or analyze a WAV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .analysis import AnalysisError, DEFAULT_FILTER, run_analysis
from .training import TrainingError, train_models
from .model_paths import LAYER1_MODEL, LAYER2_MODEL, LF_MODEL


def _ask(prompt: str, default: str) -> str:
    answer=input(f'{prompt} [{default}]: ').strip()
    return answer or default


def _ask_yes(prompt: str, default: bool = True) -> bool:
    indicator='Y/n' if default else 'y/N'
    while True:
        value=input(f'{prompt} [{indicator}]: ').strip().lower()
        if not value:return default
        if value in ('y','yes'):return True
        if value in ('n','no'):return False
        print('Please enter Y or N.')


def _ask_optional_cutoff() -> float | None:
    """Blank selects the Layer 1 predicted HP cutoff."""
    while True:
        value = input('LF isolation LP cutoff in Hz (blank = automatic): ').strip()
        if not value:
            return None
        try:
            cutoff = float(value)
            if 0 < cutoff < 24000 and __import__('math').isfinite(cutoff):
                return cutoff
        except ValueError:
            pass
        print('Enter a finite number between 0 and 24000 Hz, or leave blank.')



def _select_wav() -> str:
    """Choose a bundled demo location or provide a WAV path manually."""
    options = {
        '1': 'Filters/dataset/noisy/Speaker1.wav',
        '2': '48kBounce_filt.wav',
    }
    print('\nSelect WAV recording:')
    print('1. Speaker1 (Filters/dataset/noisy/Speaker1.wav)')
    print('2. ASAPH demo (48kBounce_filt.wav in project root)')
    print('3. Specify another WAV path')
    while True:
        choice = input('Select recording [1-3] [2]: ').strip() or '2'
        if choice in options:
            wav = options[choice]
            break
        if choice == '3':
            while True:
                wav = input('WAV file path: ').strip().strip('"')
                if wav:
                    break
                print('Please enter a WAV file path.')
            break
        print('Enter 1, 2, or 3.')
    return wav

def _menu(parser) -> int:
    print('\n=== CSC-4240 AUDIO FORENSICS SYSTEM ===')
    print('1. Train models / generate full-length dataset')
    print('2. Analyze WAV recording')
    print('3. Exit')
    while True:
        choice=input('Select option [1-3]: ').strip()
        if choice in ('1','2','3'):break
        print('Enter 1, 2 or 3.')
    if choice=='3':return 0
    if choice=='1':
        audio=_ask('Directory of noisy Speaker WAVs','Filters/dataset/noisy')
        out=_ask('New training output directory','results/train-new')
        profile=_ask('Cutoff profile (realistic/unrestricted/custom)','realistic').lower()
        custom=None
        if profile=='custom':
            names=('highpass_cutoff_hz','lowpass_cutoff_hz','bandpass_low_hz','bandpass_high_hz')
            custom={}
            for name in names:
                vals=_ask(f'{name}: lower,upper Hz',','.join(map(str,({'highpass_cutoff_hz':[80,600],'lowpass_cutoff_hz':[2000,8000],'bandpass_low_hz':[80,600],'bandpass_high_hz':[2000,8000]}[name]))))
                custom[name]=[float(v.strip()) for v in vals.split(',')]
        clip=_ask('Layer 1 training window (seconds, or full)','60')
        seconds=None if clip.lower() in ('full','none','null') else float(clip)
        count=int(_ask('Layer 1 examples per recording','10'))
        l1=_ask_yes('Train Layer 1?',True)
        l2=_ask_yes('Train Layer 2 (full-band)?',True)
        lf_train=_ask_yes('Train specialized LF model?',True)
        lf_mode='standard'
        lf_window=5.0
        lf_events='layer2_noise_events.csv'
        if lf_train:
            lf_mode=_ask('LF training mode (standard/generalized)','standard').lower()
            lf_window=float(_ask('LF training window (seconds)','5'))
            lf_events=_ask('LF event annotation CSV','layer2_noise_events.csv')
        export=_ask_yes('Save 109 full-length forward/reverse WAV pairs? (large storage required)',False)
        events=_ask('Layer 2 Ardour events manifest','layer2_noise_events.json') if l2 else 'layer2_noise_events.json'
        result=train_models(audio_dir=audio,out=out,profile=profile,clip_seconds=seconds,
                            examples_per_source=count,events=events,train_layer1=l1,
                            train_layer2=l2,train_lf=lf_train,lf_mode=lf_mode,
                            lf_window_seconds=lf_window,lf_events=lf_events,
                            save_full_dataset=export,custom_ranges=custom)
        print(f'Complete: {result}');return 0
    wav=_select_wav()
    skip=_ask_yes('Skip inverse filtering (direct Layer 2 diagnostic)?',False)
    model=str(LAYER1_MODEL)
    l2=str(LAYER2_MODEL)
    lf=str(LF_MODEL) if not skip else None
    print(f'Using Layer 1 model: {model}' if not skip else 'Skipping Layer 1')
    print(f'Using Layer 2 model: {l2}')
    if lf:
        print(f'Using LF model: {lf}')
    cutoff=_ask_optional_cutoff() if lf else None
    out=_ask('New analysis output directory','results/analysis-new')
    result=run_analysis(wav,model,out,layer2_model=l2,skip_inverse=skip,
                        lf_model=lf,lf_cutoff=cutoff)
    print(f'Complete: {result}');return 0


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog='python3 -m Offline_app',description=__doc__)
    sub=parser.add_subparsers(dest='command')
    analyze=sub.add_parser('analyze',help='Layer 1 -> C inverse -> Layer 2, with optional bypass')
    analyze.add_argument('wav',help='48-kHz mono 16-bit PCM WAV')
    analyze.add_argument('--model',default=str(LAYER1_MODEL),help='Layer 1 model; default models/layer1/model.joblib')
    analyze.add_argument('--layer2-model',default=str(LAYER2_MODEL),help='Layer 2 model; default models/layer2/model.joblib')
    analyze.add_argument('--lf-model',default=str(LF_MODEL),help='LF model; default models/lf/model.joblib')
    analyze.add_argument('--no-lf',action='store_true',help='Disable specialized LF classifier')
    analyze.add_argument('--lf-cutoff',type=float,default=None,metavar='HZ',help='Override LF isolation LP cutoff; default is Layer 1 predicted HP cutoff')
    analyze.add_argument('--lf-threshold',type=float,default=0.5)
    analyze.add_argument('--lf-script',default=None,help='Path to LF_experiment/lf_experiment.py')
    analyze.add_argument('--skip-inverse',action='store_true')
    analyze.add_argument('--threshold',type=float,default=None)
    analyze.add_argument('--out',required=True)
    analyze.add_argument('--filter-bin',default=str(DEFAULT_FILTER))
    analyze.add_argument('--timeout',type=int,default=600)
    train=sub.add_parser('train',help='Train Layer 1, full-band Layer 2, specialized LF; optionally save WAV dataset')
    train.add_argument('--audio-dir',required=True,help='Speaker WAV directory')
    train.add_argument('--out',required=True,help='NEW output directory')
    train.add_argument('--events',default='layer2_noise_events.json')
    train.add_argument('--settings',default='foundational/settings.json',help='Existing Layer 1 base settings')
    train.add_argument('--profile',choices=['realistic','unrestricted','custom'],default='realistic')
    train.add_argument('--custom-ranges',help='JSON containing 4 frequency cutoff ranges (custom profile only)')
    train.add_argument('--clip-seconds',type=float,default=60)
    train.add_argument('--full-clips',action='store_true',help='Layer 1 uses whole WAVs for each example (not recommended)')
    train.add_argument('--examples-per-source',type=int,default=10)
    train.add_argument('--skip-layer1',action='store_true')
    train.add_argument('--skip-layer2',action='store_true')
    train.add_argument('--train-lf',action='store_true',help='Also train a specialized LF classifier')
    train.add_argument('--lf-mode',choices=('standard','generalized'),default='standard',
                       help='standard: one HP300/LP300 configuration (3 classes); generalized: 8 filter variants (2 classes)')
    train.add_argument('--lf-events',default='layer2_noise_events.csv',help='Ground-truth noise regions CSV for LF')
    train.add_argument('--lf-window-seconds',type=float,default=5.)
    train.add_argument('--lf-trees',type=int,default=200,help='Generalized LF Random Forest tree count')
    train.add_argument('--lf-max-speakers',type=int,default=0,help='Optional small smoke test; 0 = all available')
    train.add_argument('--save-full-dataset',action='store_true',help='Save 109 full-length forward/reverse WAV pairs')
    train.add_argument('--filter-bin',default=str(DEFAULT_FILTER))
    train.add_argument('--timeout',type=int,default=1800)
    sub.add_parser('menu',help='Interactive training/analysis menu')
    args=parser.parse_args(argv)
    try:
        if args.command in (None,'menu'):
            return _menu(parser)
        if args.command=='analyze':
            if args.timeout < 1:raise AnalysisError('--timeout must be positive')
            result=run_analysis(args.wav,args.model,args.out,args.filter_bin,timeout=args.timeout,
                               layer2_model=args.layer2_model,skip_inverse=args.skip_inverse,threshold=args.threshold,
                               lf_model=None if (args.skip_inverse or args.no_lf) else args.lf_model,
                               lf_threshold=args.lf_threshold,lf_cutoff=args.lf_cutoff,lf_script=args.lf_script)
        else:
            if args.profile=='custom' and not args.custom_ranges:
                raise TrainingError('--custom-ranges JSON required for custom profile')
            if args.custom_ranges and args.profile!='custom':
                raise TrainingError('--custom-ranges only applies to --profile custom')
            ranges=json.loads(Path(args.custom_ranges).read_text()) if args.custom_ranges else None
            result=train_models(audio_dir=args.audio_dir,out=args.out,profile=args.profile,
                               clip_seconds=None if args.full_clips else args.clip_seconds,
                               examples_per_source=args.examples_per_source,events=args.events,
                               train_layer1=not args.skip_layer1,train_layer2=not args.skip_layer2,
                               train_lf=args.train_lf,lf_mode=args.lf_mode,lf_events=args.lf_events,
                               lf_window_seconds=args.lf_window_seconds,lf_trees=args.lf_trees,
                               lf_max_speakers=args.lf_max_speakers,save_full_dataset=args.save_full_dataset,custom_ranges=ranges,
                               filter_bin=args.filter_bin,timeout=args.timeout,layer1_settings=args.settings)
        print(f'Complete: {result}')
        return 0
    except (AnalysisError,TrainingError, OSError, ValueError, TimeoutError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())

# visualization/plot_generator.py


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ── IEEE publication style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.5,
    'lines.linewidth':   1.5,
    'pdf.fonttype':      42,   # embed fonts
    'ps.fonttype':       42,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})

# ── Color palette (colorblind-friendly, IEEE-safe) ────────────────────────────
PALETTE = {
    'FP32_noDP':      '#4E79A7',
    'FP32_softDP':    '#76B7B2',
    'FedBN':          '#59A14F',
    'FedPAQ':         '#EDC948',
    'Mao_etal':       '#B07AA1',
    'BitFusion':      '#FF9DA7',
    'HMPE-ACF_noDP':  '#9C755F',
    'HMPE-ACF':       '#E15759',  # highlight
    'FedPLH':    '#E15759',
}

# Display labels
LABELS = {
    'FP32_noDP':       'FP32 (NoDP)',
    'FP32_softDP':     'FP32 (SoftDP)',
    'FedBN':           'FedBN',
    'FedPAQ':          'FedPAQ',
    'Mao_etal':        'Mao et al.',
    'BitFusion':       'BitFusion',
    'HMPE-ACF_noDP':   'FedPLH-noDP',
    'HMPE-ACF':        'FedPLH',
}

# Highlight methods (bold frame)
HIGHLIGHT = {'HMPE-ACF', 'FedPLH'}

# Latency decomposition colors
LAT_COLORS = {
    'Compute':  '#4E79A7',
    'DP_vis':   '#E15759',
    'DP_bg':    '#76B7B2',
    'Comm':     '#59A14F',
    'Agg':      '#EDC948',
    'Misc':     '#BAB0AC',
}


def _get_metric(m: dict, *keys, default=0.0):
    for k in keys:
        if k in m:
            return float(m[k])
    return default


class PlotGenerator:
    def __init__(self, output_dir: str = 'results/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 5: Convergence + Privacy meter ──────────────────────────────────
    def plot_training_curves(self, history: Dict, save_name: str = 'Fig5_Convergence'):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
        rounds = history.get('round', list(range(len(history.get('train_loss', [])))))

        c = PALETTE['HMPE-ACF']

        # (a) Val Dice
        ax = axes[0]
        for key in ['val_dice', 'val_mean_dice', 'val_acc']:
            if key in history:
                ax.plot(rounds, history[key], color=c, lw=1.5)
                break
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Val Dice (%)')
        ax.set_title('(a) Validation Metric')
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_xlim(left=0)

        # (b) Train loss
        ax = axes[1]
        if 'train_loss' in history:
            ax.plot(rounds, history['train_loss'], color=PALETTE['FedBN'], lw=1.5)
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Loss')
        ax.set_title('(b) Training Loss')
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_xlim(left=0)

        # (c) Privacy budget ε
        ax = axes[2]
        if 'epsilon' in history and history['epsilon']:
            ax.plot(rounds, history['epsilon'], color=PALETTE['FedPAQ'], lw=1.5)
            ax.set_ylabel(r'Cumulative max-client $\epsilon$')
        else:
            ax.text(0.5, 0.5, 'No privacy accounting', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_xlabel('Communication Round')
        ax.set_title('(c) Nominal RDP\nAccountant Output')
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_xlim(left=0)

        plt.tight_layout(pad=0.5)
        self._save(fig, save_name)

    # ── Fig 6: SOTA bar chart (Dice / Norm. Latency / Norm. Energy) ─────────
    def plot_sota_summary(self, results: Dict, save_name: str = 'Fig6_SOTA_Comparison'):
        print(f"Plotting SOTA Summary for: {list(results.keys())}")

        # Determine baseline
        base_key = next((k for k in ['FedBN', 'FP32_softDP', 'FP32_noDP']
                         if k in results), next(iter(results)))
        bm = results[base_key]['metrics']
        base_lat = _get_metric(bm, 'avg_latency_ms', default=1.0) or 1.0
        base_eng = _get_metric(
            bm,
            'avg_local_training_energy_mJ',
            default=1.0,
        ) or 1.0

        # Ordered scenario list (only those present)
        order = ['FP32_noDP', 'FP32_softDP', 'FedBN', 'FedPAQ',
                 'Mao_etal', 'BitFusion', 'HMPE-ACF_noDP', 'HMPE-ACF']
        keys   = [k for k in order if k in results]
        labels = [LABELS.get(k, k) for k in keys]
        colors = [PALETTE.get(k, '#888888') for k in keys]

        dice    = []
        dice_sd = []
        lat     = []
        lat_sd  = []
        eng     = []
        eng_sd  = []

        for k in keys:
            m  = results[k]['metrics']
            sd = results[k].get('metrics_std', {})
            normalized = results[k].get('normalized', {})

            d = _get_metric(m, 'test_dice', 'accuracy')
            if d > 1.0:
                dice.append(d)
            else:
                dice.append(d * 100)
            dsd = _get_metric(sd, 'test_dice', 'accuracy')
            if d > 1.0:
                dice_sd.append(dsd)
            else:
                dice_sd.append(dsd * 100)

            latency_norm = normalized.get('avg_latency_ms', {})
            if latency_norm:
                lat.append(float(latency_norm.get('mean', 0.0)))
                lat_sd.append(float(latency_norm.get('std', 0.0)))
            else:
                lat.append(_get_metric(m, 'avg_latency_ms') / base_lat)
                lat_sd.append(_get_metric(sd, 'avg_latency_ms') / base_lat)

            e_val = _get_metric(
                m,
                'avg_local_training_energy_mJ',
            )
            energy_norm = normalized.get('avg_local_training_energy_mJ', {})
            if energy_norm:
                eng.append(float(energy_norm.get('mean', 0.0)))
                eng_sd.append(float(energy_norm.get('std', 0.0)))
            else:
                eng.append(e_val / base_eng)
                eng_sd.append(
                    _get_metric(
                        sd,
                        'avg_local_training_energy_mJ',
                    ) / base_eng
                )

        n   = len(keys)
        x   = np.arange(n)
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.05))

        def _bar(ax, vals, errs, ylabel, ylim=None, fmt='{:.1f}', highlight_top=True):
            bars = ax.bar(x, vals, yerr=errs, capsize=2,
                          color=colors, edgecolor='white',
                          linewidth=0.5, error_kw={'lw': 0.8, 'ecolor': '#444'})
            # Bold edge for highlight methods
            for i, k in enumerate(keys):
                if k in HIGHLIGHT:
                    bars[i].set_edgecolor('#222222')
                    bars[i].set_linewidth(1.2)
            # Value annotations
            for i, (v, e) in enumerate(zip(vals, errs)):
                ax.text(i, v + e + (ylim[1] - ylim[0]) * 0.01 if ylim else v + e * 1.1,
                        fmt.format(v), ha='center', va='bottom',
                        fontsize=6.0, color='#222')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6.5, rotation=32, ha='right')
            ax.set_ylabel(ylabel, fontsize=7.5)
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, axis='y', ls='--', alpha=0.4, lw=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ymin_dice = max(0, min(d - e for d, e in zip(dice, dice_sd)) - 2)
        _bar(axes[0], dice, dice_sd, 'Dice (%)',
             ylim=(ymin_dice, max(d + e for d, e in zip(dice, dice_sd)) + 4),
             fmt='{:.1f}')
        axes[0].set_title('(a) Dice score', fontsize=8.5, pad=9)

        _bar(axes[1], lat, lat_sd, 'Norm. latency',
             ylim=(0, max(v + e for v, e in zip(lat, lat_sd)) * 1.25), fmt='{:.2f}')
        axes[1].axhline(1.0, color='gray', ls=':', lw=0.8)
        axes[1].set_title('(b) Normalized latency', fontsize=8.5, pad=9)

        _bar(axes[2], eng, eng_sd,
             'Normalized modeled local-training\ncompute/memory energy',
             ylim=(0, max(v + e for v, e in zip(eng, eng_sd)) * 1.25), fmt='{:.2f}')
        axes[2].axhline(1.0, color='gray', ls=':', lw=0.8)
        axes[2].set_title(
            '(c) Modeled local-training compute/memory energy',
            fontsize=8.5,
            pad=9,
        )

        fig.subplots_adjust(
            left=0.075,
            right=0.995,
            bottom=0.34,
            top=0.82,
            wspace=0.48,
        )
        self._save(fig, save_name)

    # ── BEU latency breakdown (论文 Fig6) ────────────────────────────────────
    def plot_beu_latency_breakdown(self, results: Dict,
                                   save_name: str = 'Fig6_BEU_Latency_Breakdown'):
        """
        单轮时延分解：Compute / DP_visible / DP_bg / Comm / Agg
        对比 FP32_noDP, FP32_softDP, HMPE-ACF
        """
        target_keys = ['FP32_noDP', 'FP32_softDP', 'HMPE-ACF']
        present = [k for k in target_keys if k in results]
        if not present:
            print(f"  Skip BEU breakdown: none of {target_keys} in results")
            return

        x_labels = [LABELS.get(k, k) for k in present]
        n = len(present)

        compute_vals, dp_vis_vals, dp_bg_vals = [], [], []
        comm_vals, agg_vals, misc_vals, total_vals = [], [], [], []
        for k in present:
            m = results[k]['metrics']
            total_lat    = _get_metric(m, 'avg_latency_ms')
            dp_overhead  = _get_metric(m, 'avg_dp_overhead_ms')
            dp_bg        = _get_metric(
                m,
                'avg_dp_background_ms',
                default=max(0.0, _get_metric(m, 'avg_dp_total_ms') - dp_overhead),
            )
            comm         = _get_metric(m, 'avg_comm_latency_ms')
            agg          = _get_metric(m, 'avg_agg_latency_ms')
            misc         = _get_metric(m, 'avg_misc_latency_ms')
            compute      = _get_metric(
                m,
                'avg_compute_latency_ms',
                default=max(0.0, total_lat - dp_overhead - comm - agg - misc),
            )
            reconstructed = compute + dp_overhead + comm + agg + misc
            if abs(reconstructed - total_lat) > 1e-6:
                raise ValueError(
                    f"Latency breakdown for {k} does not close: "
                    f"{reconstructed} != {total_lat}"
                )

            compute_vals.append(compute)
            dp_vis_vals.append(dp_overhead)
            dp_bg_vals.append(dp_bg)
            comm_vals.append(comm)
            agg_vals.append(agg)
            misc_vals.append(misc)
            total_vals.append(total_lat)

        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        x = np.arange(n)
        width = 0.55

        b1 = ax.bar(x, compute_vals, width, label='Compute',
                    color=LAT_COLORS['Compute'], edgecolor='white', lw=0.5)
        b2 = ax.bar(x, dp_vis_vals, width, bottom=compute_vals,
                    label='Visible SoftDP', color=LAT_COLORS['DP_vis'],
                    edgecolor='white', lw=0.5)
        b4 = ax.bar(x, comm_vals, width,
                    bottom=[c + v for c, v in zip(compute_vals, dp_vis_vals)],
                    label='Comm', color=LAT_COLORS['Comm'],
                    edgecolor='white', lw=0.5)
        b5 = ax.bar(x, agg_vals, width,
                    bottom=[c + v + cm for c, v, cm in
                            zip(compute_vals, dp_vis_vals, comm_vals)],
                    label='Agg', color=LAT_COLORS['Agg'],
                    edgecolor='white', lw=0.5)
        if any(value > 0 for value in misc_vals):
            ax.bar(
                x,
                misc_vals,
                width,
                bottom=[
                    c + v + cm + a
                    for c, v, cm, a in zip(
                        compute_vals,
                        dp_vis_vals,
                        comm_vals,
                        agg_vals,
                    )
                ],
                label='Misc',
                color=LAT_COLORS['Misc'],
                edgecolor='white',
                lw=0.5,
            )

        for i in range(n):
            total = total_vals[i]
            annotation = f'{total:.0f} ms'
            if dp_bg_vals[i] > 0:
                annotation += (
                    f'\nbudget-covered\nSoftDP: {dp_bg_vals[i]:.1f} ms'
                )
            ax.text(i, total + max(total_vals) * 0.02, annotation,
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel('Avg per-round latency (ms)', fontsize=8)
        ax.set_title('Critical-path latency decomposition\n(mean over five seeds)', fontsize=8)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8,
                  ncol=1, handlelength=1.2)
        ax.grid(True, axis='y', ls='--', alpha=0.4, lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(total_vals) * 1.25)

        plt.tight_layout(pad=0.5)
        self._save(fig, save_name)

    # ── Fig 7: SAC scalability ────────────────────────────────────────────────
    def plot_scalability_analysis(self, scalability: Dict,
                                  save_name: str = 'Fig7_Scalability'):
        """SAC vs CPU aggregation latency as K grows."""
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))

        clients = sorted([int(k) for k in scalability.get('clients', {}).keys()])
        if not clients:
            print("  Skip scalability: no 'clients' key in data")
            plt.close(fig)
            return

        model_sizes = ['10MiB', '50MiB', '100MiB']
        colors_sac = ['#4E79A7', '#E15759', '#59A14F']
        colors_cpu = ['#A0CBE8', '#F28E2B', '#8CD17D']

        ax = axes[0]
        ax.set_title('(a) Fixed-resource aggregation latency', fontsize=8)
        for i, ms in enumerate(model_sizes):
            sac_lat, cpu_lat = [], []
            for c in clients:
                cd = scalability['clients'].get(str(c), {})
                sac_lat.append(cd.get(f'sac_{ms.lower()}', cd.get('sac', 0)))
                cpu_lat.append(cd.get(f'cpu_{ms.lower()}', cd.get('cpu', 0)))
            if any(v > 0 for v in sac_lat):
                ax.plot(clients, sac_lat, 'o-', color=colors_sac[i],
                        lw=1.4, ms=4, label=f'SAC ({ms})')
            if any(v > 0 for v in cpu_lat):
                ax.plot(clients, cpu_lat, 's--', color=colors_cpu[i],
                        lw=1.4, ms=4, label=f'CPU ({ms})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Clients', fontsize=8)
        ax.set_ylabel('Aggregation Latency (ms)', fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = axes[1]
        ax.set_title('(b) Modeled SAC speedup', fontsize=8)
        for i, ms in enumerate(model_sizes):
            speedups = []
            for c in clients:
                cd = scalability['clients'].get(str(c), {})
                sac = cd.get(f'sac_{ms.lower()}', cd.get('sac', 1))
                cpu = cd.get(f'cpu_{ms.lower()}', cd.get('cpu', 1))
                speedups.append(cpu / sac if sac > 0 else 0)
            if any(v > 0 for v in speedups):
                ax.plot(clients, speedups, 'o-', color=colors_sac[i],
                        lw=1.4, ms=4, label=ms)
                # Annotate max
                mx = max(speedups)
                mxi = speedups.index(mx)
                ax.annotate(f'{mx:.2f}× @ {clients[mxi]} clients',
                            xy=(clients[mxi], mx),
                            xytext=(clients[mxi] * 0.5, mx * 0.85),
                            fontsize=6.5, color=colors_sac[i],
                            arrowprops=dict(arrowstyle='->', color=colors_sac[i], lw=0.8))
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='No speedup')
        ax.set_xscale('log')
        ax.set_xlabel('Number of Clients', fontsize=8)
        ax.set_ylabel('Speedup (SAC vs CPU)', fontsize=8)
        ax.legend(fontsize=6.5)
        ax.grid(True, which='both', ls='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout(pad=0.6)
        self._save(fig, save_name)

    # ── generate_all_figures ─────────────────────────────────────────────────
    def generate_all_figures(self, results_dir: str = 'results'):
        results_path = Path(results_dir)

        print("=" * 72)
        print("GENERATING ALL PAPER FIGURES")
        print(f"Results dir: {results_path.resolve()}")
        print(f"Output dir:  {self.output_dir.resolve()}")
        print("=" * 72)

        # ── Fig5: Convergence ──────────────────────────────────────────────
        hist_path = results_path / "summaries" / "representative_history.json"
        if not hist_path.exists():
            hist_path = results_path / "training_history.json"
        if hist_path.exists():
            with open(hist_path, encoding='utf-8') as f:
                history = json.load(f)
            self.plot_training_curves(history, save_name="Fig5_Convergence")
            print("Fig5_Convergence generated")
        else:
            print("Skip Fig5: training_history.json not found")

        # ── Fig6: SOTA bar chart ───────────────────────────────────────────
        sota_path = results_path / "summaries" / "paper_results.json"
        if not sota_path.exists():
            sota_path = results_path / "summaries" / "sota_results.json"
        if sota_path.exists():
            with open(sota_path, encoding='utf-8') as f:
                payload = json.load(f)
            results = payload.get("results", payload)
            self.plot_sota_summary(results, save_name="Fig6_SOTA_Comparison")
            print("Fig6_SOTA_Comparison generated")
            # BEU latency breakdown
            self.plot_beu_latency_breakdown(results, save_name="Fig6b_BEU_Breakdown")
            print("Fig6b_BEU_Breakdown generated")
        else:
            print("Skip Fig6: ablation/ablation_results.json not found")

        # ── Fig7: SAC scalability ──────────────────────────────────────────
        scal_path = results_path / "scalability" / "scalability_results.json"
        if scal_path.exists():
            with open(scal_path, encoding='utf-8') as f:
                scalability = json.load(f)
            self.plot_scalability_analysis(scalability, save_name="Fig7_Scalability")
            print("Fig7_Scalability generated")
        else:
            print("Skip Fig7: scalability/scalability_results.json not found")

        print("=" * 72)
        print(f"All available figures saved to: {self.output_dir}")
        print("=" * 72)

    def _save(self, fig, name: str):
        for ext in ['pdf', 'png']:
            fig.savefig(self.output_dir / f'{name}.{ext}')
        plt.close(fig)

    # Keep old name aliases for backward compatibility
    def plot_ablation_comparison(self, *args, **kwargs):
        pass

    def plot_privacy_tradeoff(self, *args, **kwargs):
        pass

    def plot_roofline_model(self, *args, **kwargs):
        pass

    def plot_precision_comparison(self, *args, **kwargs):
        pass

    def _save_figure(self, fig, name: str):
        self._save(fig, name)

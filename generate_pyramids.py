"""
MST Population Pyramid Generator — Colab-ready
Upload mst_audit_gpt-image-1-mini.csv and mst_audit_grok-imagine-image.csv
to the same directory, then run this script.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

# Monk Skin Tone scale colors (1=lightest, 10=darkest)
MST_COLORS = {
    1: '#F6EDE4', 2: '#F3E7DB', 3: '#F7D7C4', 4: '#EACEB0', 5: '#D7B89C',
    6: '#A07860', 7: '#895B40', 8: '#6C4030', 9: '#3F2218', 10: '#2D1A10',
}

def load_data(filepath):
    """Load L5 (G+O+I+S+N) data, split by income × setting."""
    high_city, high_village = [], []
    low_city, low_village = [], []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['cues'].strip() != 'G+O+I+S+N':
                continue
            income = row['income'].strip().lower()
            setting = row['setting'].strip().lower()
            if income not in ('high', 'low') or setting not in ('city', 'village'):
                continue
            mst = int(row['MST_value_p1'])
            if income == 'high':
                (high_city if setting == 'city' else high_village).append(mst)
            else:
                (low_city if setting == 'city' else low_village).append(mst)
    return high_city, high_village, low_city, low_village

def count_mst(values):
    counts = {i: 0 for i in range(1, 11)}
    for v in values:
        if 1 <= v <= 10:
            counts[v] += 1
    return counts

def draw_pyramid(ax, high_city, high_village, low_city, low_village, title):
    mst_range = range(1, 11)
    hc = count_mst(high_city)
    hv = count_mst(high_village)
    lc = count_mst(low_city)
    lv = count_mst(low_village)

    bar_h = 0.7

    for mst_val in mst_range:
        color = MST_COLORS[mst_val]
        ec = '#555555' if mst_val <= 6 else '#222222'

        # Overlapping bars: draw larger first (behind), smaller on top
        # HIGH INCOME side (negative direction)
        h_city_val = hc[mst_val]
        h_vill_val = hv[mst_val]
        if h_vill_val >= h_city_val:
            ax.barh(mst_val, -h_vill_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6, hatch='////', alpha=0.75)
            ax.barh(mst_val, -h_city_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6)
        else:
            ax.barh(mst_val, -h_city_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6)
            ax.barh(mst_val, -h_vill_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6, hatch='////', alpha=0.75)

        # LOW INCOME side (positive direction)
        l_city_val = lc[mst_val]
        l_vill_val = lv[mst_val]
        if l_vill_val >= l_city_val:
            ax.barh(mst_val, l_vill_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6, hatch='////', alpha=0.75)
            ax.barh(mst_val, l_city_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6)
        else:
            ax.barh(mst_val, l_city_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6)
            ax.barh(mst_val, l_vill_val, height=bar_h, color=color,
                    edgecolor=ec, linewidth=0.6, hatch='////', alpha=0.75)

    ax.set_yticks(list(mst_range))
    ax.set_yticklabels(list(mst_range), fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1.2)

    all_max = max(
        max(hc[m] for m in mst_range), max(hv[m] for m in mst_range),
        max(lc[m] for m in mst_range), max(lv[m] for m in mst_range),
    ) + 1
    limit = int(all_max * 1.25) + 3
    ax.set_xlim(-limit, limit)

    raw_ticks = np.linspace(-limit, limit, 9).astype(int)
    ax.set_xticks(raw_ticks)
    ax.set_xticklabels([str(abs(t)) for t in raw_ticks], fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Monk Skin Tone Scale (MST) Labels', fontsize=10)
    ax.set_xlabel('Count of Generated Images', fontsize=10)

    ax.text(-limit * 0.5, 0.15, 'High Income', ha='center', fontsize=11,
            fontweight='bold', fontstyle='italic')
    ax.text(limit * 0.5, 0.15, 'Low Income', ha='center', fontsize=11,
            fontweight='bold', fontstyle='italic')
    ax.text(-limit * 0.92, 0.75, 'Lighter Skin', ha='left', fontsize=8,
            fontstyle='italic', color='gray')
    ax.text(-limit * 0.92, 10.3, 'Darker Skin', ha='left', fontsize=8,
            fontstyle='italic', color='gray')


# ── Paths (adjust if your CSVs are elsewhere) ──
GPT_CSV = 'mst_audit_gpt-image-1-mini.csv'
GROK_CSV = 'mst_audit_grok-imagine-image.csv'

gpt = load_data(GPT_CSV)
grok = load_data(GROK_CSV)

city_patch = mpatches.Patch(facecolor='#D7B89C', edgecolor='#555', label='City Prompt')
village_patch = mpatches.Patch(facecolor='#D7B89C', edgecolor='#555', hatch='////',
                                alpha=0.75, label='Village Prompt')

# ── Figure 1: GPT ──
fig1, ax1 = plt.subplots(figsize=(10, 7))
draw_pyramid(ax1, *gpt, 'MST Distribution across Prompts (GPT-image-1-mini)')
ax1.legend(handles=[city_patch, village_patch], loc='lower right', fontsize=11,
           frameon=True, title='Setting', title_fontsize=11)
fig1.text(0.5, -0.02,
          'Population-pyramid histogram: MST distribution of high/low-income prompts\n'
          'split by setting (city vs village) for GPT-image-1-mini.',
          ha='center', fontsize=10, fontstyle='italic')
plt.tight_layout()
fig1.savefig('mst_pyramid_gpt.png', bbox_inches='tight', facecolor='white', dpi=200, pad_inches=0.3)
plt.show()
plt.close(fig1)

# ── Figure 2: Grok ──
fig2, ax2 = plt.subplots(figsize=(10, 7))
draw_pyramid(ax2, *grok, 'MST Distribution across Prompts (Grok-imagine)')
ax2.legend(handles=[city_patch, village_patch], loc='lower right', fontsize=11,
           frameon=True, title='Setting', title_fontsize=11)
fig2.text(0.5, -0.02,
          'Population-pyramid histogram: MST distribution of high/low-income prompts\n'
          'split by setting (city vs village) for Grok-imagine.',
          ha='center', fontsize=10, fontstyle='italic')
plt.tight_layout()
fig2.savefig('mst_pyramid_grok.png', bbox_inches='tight', facecolor='white', dpi=200, pad_inches=0.3)
plt.show()
plt.close(fig2)

print("Done — saved mst_pyramid_gpt.png and mst_pyramid_grok.png")

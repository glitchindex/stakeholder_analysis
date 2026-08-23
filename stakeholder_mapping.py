"""
Stakeholder relationship mapping.

Expects a CSV (Line 129 - format available on GitHub /glitchindex) with these columns (header text is matched flexibly, but
each field is identified by whole-word tokens so similarly-named columns
- e.g. POWER/INFLUENCE vs INFLUENCES vs INFLUENCED BY - never collide):

    STAKEHOLDER        name / position
    RACI               Responsible / Accountable / Consulted / Informed
    POWER/INFLUENCE    1-5
    URGENCY            1-5
    INTEREST           1-5
    RISK/REWARD        free text (not visualized - see NOTE at bottom)
    STANCE             Champion / Indifferent / Skeptical / Blocking
    TIER               Core / Satellite
    INFLUENCES         semicolon-separated names this person influences
    INFLUENCED BY      semicolon-separated names who influence this person

Visual encoding:
    Node fill color   -> Tier (Core / Satellite)
    Node size         -> Power/Influence (1-5)
    Node border style -> RACI (solid/dashed/dotted, thickness)
    Node border color -> Urgency (pale yellow -> deep red)
    Halo behind node  -> Interest (bigger halo = more interested)
    Marker shape      -> Stance (star / circle / triangle / X)
    Edge direction    -> taken directly from INFLUENCES / INFLUENCED BY
    Edge color        -> tier of the influencing (source) stakeholder
"""

import re
import warnings

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

### Visual Encoding

TIER_COLORS = {"Core": "#FF0055", "Satellite": "#006B7B"}
DEFAULT_TIER = "Core"

RACI_STYLES = {  # raci -> (linestyle, linewidth)
    "Accountable": ("solid", 3.5),
    "Responsible": ("solid", 1.4),
    "Consulted": ("dashed", 2.2),
    "Informed": ("dotted", 2.2),
}
DEFAULT_RACI = "Informed"

STANCE_MARKERS = {
    "Champion": "*",
    "Indifferent": "o",
    "Skeptical": "^",
    "Blocking": "X",
}
DEFAULT_STANCE = "Indifferent"

URGENCY_CMAP = plt.cm.YlOrRd  # pale yellow (low urgency) -> deep red (high)
POWER_SIZE_MIN, POWER_SIZE_MAX = 700, 2900
HALO_MIN, HALO_MAX = 0, 2400


### Detection/Formatting

def _tokens(col):
    cleaned = col.strip().strip("\ufeff").upper()
    return set(re.split(r"[^A-Z0-9]+", cleaned)) - {""}


def _find_column(df, token_sets, exclude_tokens=None, label="column"):
    
    exclude_tokens = exclude_tokens or set()
    for col in df.columns:
        tokens = _tokens(col)
        if tokens & exclude_tokens:
            continue
        for candidate in token_sets:
            if candidate <= tokens:
                return col
    warnings.warn(
        f"Could not confidently identify the '{label}' column - "
        f"treating it as missing rather than guessing."
    )
    return None


def _split_names(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [n.strip() for n in str(value).split(";") if n.strip()]


def _match_node(name, node_names, context=""):
    if not name:
        return None
    lower_map = {n.lower(): n for n in node_names}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    candidates = [
        n for n in node_names
        if name.lower() in n.lower() or n.lower() in name.lower()
    ]
    if len(candidates) == 1:
        print(f"Note: matched '{name}' to '{candidates[0]}' by partial match{context}.")
        return candidates[0]
    if len(candidates) > 1:
        print(f"Warning: '{name}'{context} matches multiple stakeholders {candidates} - skipping.")
        return None
    print(f"Warning: could not find a stakeholder matching '{name}'{context} - skipping.")
    return None


def _to_scale(value, low=1, high=5, default=3):
    try:
        return min(max(float(value), low), high)
    except (TypeError, ValueError):
        return default


def _scaled(value, low=1, high=5, out_min=0, out_max=1):
    return out_min + (value - low) / (high - low) * (out_max - out_min)


### Main Body

def generate_stakeholder_map(csv_file="stakeholders.csv", output_img="stakeholder_network.png"):
    try:
        df = pd.read_csv(csv_file, sep=",", encoding="utf-8-sig")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    df.columns = [c.strip().strip("\ufeff") for c in df.columns]

    col_name = _find_column(df, [{"STAKEHOLDER"}, {"NAME"}, {"PERSON"}], label="stakeholder name")
    if col_name is None:
        print("Could not find a stakeholder name column - aborting.")
        return

    col_raci = _find_column(df, [{"RACI"}], label="RACI")
    col_power = _find_column(df, [{"POWER"}], label="power/influence")
    col_urgency = _find_column(df, [{"URGENCY"}], label="urgency")
    col_interest = _find_column(df, [{"INTEREST"}], label="interest")
    col_stance = _find_column(df, [{"STANCE"}, {"SUPPORT"}], label="stance")
    col_tier = _find_column(df, [{"TIER"}, {"RING"}], label="tier")
    col_influences = _find_column(df, [{"INFLUENCES"}], exclude_tokens={"BY"}, label="influences (outgoing)")
    col_influenced_by = _find_column(df, [{"INFLUENCED", "BY"}], label="influenced by (incoming)")

    # 1. Nodes
    G = nx.DiGraph()
    for _, row in df.iterrows():
        name = str(row[col_name]).strip()
        if not name or name.lower() == "nan":
            continue

        tier = str(row[col_tier]).strip().title() if col_tier else DEFAULT_TIER
        raci = str(row[col_raci]).strip().title() if col_raci else DEFAULT_RACI
        stance = str(row[col_stance]).strip().title() if col_stance else DEFAULT_STANCE

        G.add_node(
            name,
            tier=tier if tier in TIER_COLORS else DEFAULT_TIER,
            raci=raci if raci in RACI_STYLES else DEFAULT_RACI,
            stance=stance if stance in STANCE_MARKERS else DEFAULT_STANCE,
            power=_to_scale(row[col_power]) if col_power else 3,
            urgency=_to_scale(row[col_urgency]) if col_urgency else 3,
            interest=_to_scale(row[col_interest]) if col_interest else 3,
        )

    node_names = list(G.nodes())
    if not node_names:
        print("No stakeholders found - aborting.")
        return

    # 2. Edges - direction comes straight from the data, no inference required
    influences_map, influenced_by_map = {}, {}
    for _, row in df.iterrows():
        source = str(row[col_name]).strip()
        if source not in G:
            continue
        influences_map.setdefault(source, set())
        influenced_by_map.setdefault(source, set())

        if col_influences:
            for raw in _split_names(row[col_influences]):
                t = _match_node(raw, node_names, f" (from {source}'s INFLUENCES)")
                if t and t != source:
                    influences_map[source].add(t)
                    G.add_edge(source, t)

        if col_influenced_by:
            for raw in _split_names(row[col_influenced_by]):
                t = _match_node(raw, node_names, f" (from {source}'s INFLUENCED BY)")
                if t and t != source:
                    influenced_by_map[source].add(t)
                    G.add_edge(t, source)

    # 3. Consistency check - the two columns are filled in independently by different people/rows and can drift out of sync - handled by flagging.
    if col_influences and col_influenced_by:
        for person, targets in influences_map.items():
            for target in targets:
                if person not in influenced_by_map.get(target, set()):
                    print(
                        f"Note: {person} lists INFLUENCES {target}, but {target}'s "
                        f"INFLUENCED BY doesn't list {person} - confirm this is intentional."
                    )

    # 4. Layout - concentric shells by tier
    core_nodes = [n for n, d in G.nodes(data=True) if d["tier"] == "Core"]
    satellite_nodes = [n for n, d in G.nodes(data=True) if d["tier"] != "Core"]
    if not core_nodes or not satellite_nodes:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.shell_layout(G, nlist=[core_nodes, satellite_nodes])

    # 5. Render
    fig = plt.figure(figsize=(15, 12), dpi=300)
    ax = plt.gca()

    # Group by stance since marker shape is set per draw call, not per node
    by_stance = {}
    for n in G.nodes():
        by_stance.setdefault(G.nodes[n]["stance"], []).append(n)

    for stance, nodelist in by_stance.items():
        marker = STANCE_MARKERS.get(stance, DEFAULT_STANCE)
        colors = [TIER_COLORS[G.nodes[n]["tier"]] for n in nodelist]
        sizes = [
            POWER_SIZE_MIN + _scaled(G.nodes[n]["power"], out_max=1) * (POWER_SIZE_MAX - POWER_SIZE_MIN)
            for n in nodelist
        ]
        edge_colors = [URGENCY_CMAP(_scaled(G.nodes[n]["urgency"])) for n in nodelist]
        linewidths = [RACI_STYLES[G.nodes[n]["raci"]][1] for n in nodelist]
        linestyles = [RACI_STYLES[G.nodes[n]["raci"]][0] for n in nodelist]
        halo_sizes = [
            HALO_MIN + _scaled(G.nodes[n]["interest"], out_max=1) * (HALO_MAX - HALO_MIN)
            for n in nodelist
        ]

        # Halo (Interest) drawn behind the node
        xs = [pos[n][0] for n in nodelist]
        ys = [pos[n][1] for n in nodelist]
        ax.scatter(xs, ys, s=[s0 + h for s0, h in zip(sizes, halo_sizes)],
                   c=colors, alpha=0.15, linewidths=0, zorder=1)

        nodes_collection = nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist, node_color=colors, node_size=sizes,
            node_shape=marker, edgecolors=edge_colors, linewidths=linewidths,
            alpha=0.95, ax=ax,
        )
        if nodes_collection is not None:
            nodes_collection.set_linestyle(linestyles)
            nodes_collection.set_zorder(3)

    edge_colors = [TIER_COLORS[G.nodes[u]["tier"]] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True, arrowsize=16,
        arrowstyle="-|>", node_size=1800, connectionstyle="arc3,rad=0.08",
        width=1.4, alpha=0.6, ax=ax,
    )

    labels = {n: n.replace(" (", "\n(") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold",
                             font_color="#111827", ax=ax)

    ### Legend
    tier_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=11, label=t)
        for t, c in TIER_COLORS.items()
    ]
    stance_handles = [
        mlines.Line2D([0], [0], marker=m, color="w", markerfacecolor="#475569", markersize=11, label=s)
        for s, m in STANCE_MARKERS.items()
    ]
    raci_handles = [
        mlines.Line2D([0], [0], color="#334155", linestyle=ls, linewidth=lw, label=r)
        for r, (ls, lw) in RACI_STYLES.items()
    ]
    misc_handles = [
        mpatches.Patch(color="none", label="Size = Power/Influence (1-5)"),
        mpatches.Patch(color="none", label="Halo size = Interest (1-5)"),
        mpatches.Patch(color="none", label="Border color = Urgency (pale->red)"),
    ]

    legend1 = ax.legend(handles=tier_handles + stance_handles, loc="upper left",
                         title="Tier / Stance", fontsize=7, title_fontsize=8, framealpha=0.9)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=raci_handles, loc="lower left",
                         title="RACI (border style)", fontsize=7, title_fontsize=8, framealpha=0.9)
    ax.add_artist(legend2)
    ax.legend(handles=misc_handles, loc="lower right", fontsize=7, framealpha=0.9,
              handlelength=0, handletextpad=0)

    plt.title("Stakeholder Influence Network", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_img, bbox_inches="tight")
    print(f"Network graph generated and saved to {output_img}")
    plt.close(fig)


if __name__ == "__main__":
    generate_stakeholder_map()

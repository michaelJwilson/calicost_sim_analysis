from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score

from calicost import arg_parse


def read_gene_loc(hg_table_file):
    """
    Read in (chronological order, start, end) table for (coding) gene definition.
    """
    df_hgtable = pd.read_csv(hg_table_file, sep="\t", header=0, index_col=0)

    df_hgtable = df_hgtable[df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    # NB add chr column as integer without "chr" prefix
    df_hgtable["chr"] = [int(x[3:]) for x in df_hgtable.chrom]
    df_hgtable = df_hgtable.rename(
        columns={"cdsStart": "start", "cdsEnd": "end", "name2": "gene"}
    )
    df_hgtable.set_index("gene", inplace=True)

    return df_hgtable[["chr", "start", "end"]]


def get_config_path(calico_pure_dir, sampleid):
    return f"{calico_pure_dir}/{sampleid}/configfile0"


def get_config(configuration_file, verbose=False):
    """
    Retrieve the CalicoST config.
    """
    config = None

    if not Path(configuration_file).exists():
        return None

    if verbose:
        print(f"Reading configuration file: {configuration_file}")

    try:
        config = arg_parse.read_configuration_file(configuration_file)
    except:
        print(f"Error reading as single configuration, {configuration_file}")

        try:
            config = arg_parse.read_joint_configuration_file(configuration_file)
        except:
            print(f"Error reading as joint configuration, {configuration_file}")
            return None

    if verbose:
        for key in config:
            print(f"{key}: {config[key]}")

    return config


def get_rdrbaf(
    configuration_file,
    random_state,
    relative_path="../nomixing_calicost_related/",
    verbose=False,
):
    """
    Retrieve the CalicoST RDR/BAF determinations.
    """
    config = get_config(configuration_file)

    output_dir = relative_path + config["output_dir"].split("/")[-1]
    outdir = f"{output_dir}/clone{config['n_clones']}_rectangle{random_state}_w{config['spatial_weight']:.1f}"

    fpath = f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz"

    if Path(fpath).exists():
        res_combine = dict(
            np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz"),
            allow_pickle=True,
        )

        return res_combine
    else:
        if verbose:
            print(f"CalicoST RDR/BAF determinations do not exist for {fpath}")

        return None


def get_best_r_hmrf(
    configuration_file, relative_path="../nomixing_calicost_related/", verbose=False
):
    """
    Retrieve the CalicoST random initialization with the maximum likelihood.
    """
    config = get_config(configuration_file, verbose=verbose)

    # NB find the best HMRF initialization random seed
    df_clone = []

    for random_state in range(10):
        rdrbaf = get_rdrbaf(
            configuration_file, random_state, relative_path=relative_path
        )

        if rdrbaf is not None:
            df_clone.append(
                pd.DataFrame(
                    {
                        "random seed": random_state,
                        "log-likelihood": rdrbaf["total_llf"],
                    },
                    index=[0],
                )
            )

    df_clone = pd.concat(df_clone, ignore_index=True)
    idx = np.argmax(df_clone["log-likelihood"])

    r_hmrf_initialization = df_clone["random seed"].iloc[idx]

    return int(r_hmrf_initialization)


def read_true_gene_cna(df_hgtable, truth_cna_file):
    """
    Read true copy number aberrations
    """
    true_gene_cna = df_hgtable.copy()
    df_cna = pd.read_csv(truth_cna_file, header=0, index_col=0, sep="\t")

    unique_clones = df_cna.index.unique()

    for clonename in unique_clones:
        clonal_cna = df_cna[df_cna.index == clonename]

        gene_status_str = np.array(["1|1"] * true_gene_cna.shape[0])

        # NB NEUTRAL == (1,1).
        gene_status_type = np.array(["NEU"] * true_gene_cna.shape[0], dtype="<U5")

        for i in range(clonal_cna.shape[0]):
            cna_str = f"{clonal_cna.A_copy.values[i]}|{clonal_cna.B_copy.values[i]}"

            if clonal_cna.A_copy.values[i] + clonal_cna.B_copy.values[i] > 2:
                cna_type = "AMP"
            elif clonal_cna.A_copy.values[i] + clonal_cna.B_copy.values[i] < 2:
                cna_type = "DEL"
            else:
                # TODO BUG (2,0) == CNLOH.
                cna_type = "CNLOH"

            is_affected = true_gene_cna.chr == clonal_cna.chr.values[i]

            # TODO NB just-touch criteria -> significant overlap.
            is_affected &= true_gene_cna.end >= clonal_cna.start.values[i]
            is_affected &= true_gene_cna.start <= clonal_cna.end.values[i]

            gene_status_str[is_affected] = cna_str
            gene_status_type[is_affected] = cna_type

        true_gene_cna[f"{clonename}_str"] = gene_status_str
        true_gene_cna[f"{clonename}_type"] = gene_status_type

    return true_gene_cna


def get_calico_cna_file(configuration_file):
    r_calico = get_best_r_hmrf(configuration_file)

    tmpdir = "/".join(configuration_file.split("/")[:-1])

    return f"{tmpdir}/clone3_rectangle{r_calico}_w1.0/cnv_genelevel.tsv"


def read_calico_gene_cna(configuration_file):
    """
    Read CalicoST estimated copy number aberrations.
    """
    calico_cna_file = get_calico_cna_file(configuration_file)

    calico_gene_cna = pd.read_csv(
        calico_cna_file,
        header=0,
        index_col=0,
        sep="\t",
    )

    calico_clones = (
        calico_gene_cna.columns[calico_gene_cna.columns.str.endswith("A")]
        .str.split(" ")
        .str[0]
    )

    for c in calico_clones:
        cna_type_assay = np.array(["NEU"] * calico_gene_cna.shape[0], dtype="<U5")

        cna_type_assay[
            calico_gene_cna[f"{c} A"].values + calico_gene_cna[f"{c} B"].values < 2
        ] = "DEL"

        cna_type_assay[
            calico_gene_cna[f"{c} A"].values + calico_gene_cna[f"{c} B"].values > 2
        ] = "AMP"

        is_cnloh = (
            calico_gene_cna[f"{c} A"].values + calico_gene_cna[f"{c} B"].values == 2
        ) & (calico_gene_cna[f"{c} A"].values != calico_gene_cna[f"{c} B"].values)

        cna_type_assay[is_cnloh] = "CNLOH"

        calico_gene_cna[f"{c}_type"] = cna_type_assay

    return calico_gene_cna


def read_numbat_gene_cna(bulk_clones_final_file):
    df_numbat = pd.read_csv(bulk_clones_final_file, header=0, sep="\t")
    df_numbat = df_numbat[["gene", "sample", "cnv_state"]]

    n_numbat_samples = len(np.unique(df_numbat["sample"]))

    # NB map numbat state to DEL, NEU, AMP, CNLOH
    mapper = {
        "del": "DEL",
        "bdel": "DEL",
        "neu": "NEU",
        "amp": "AMP",
        "bamp": "AMP",
        "loh": "CNLOH",
    }
    df_numbat["cnv_state"] = df_numbat["cnv_state"].replace(mapper)

    # NB add "clone" prefix to numbat clone
    df_numbat["sample"] = "clone_" + df_numbat["sample"].astype(str) + "_type"

    # NB reshape the table such that each row is a gene and each column is a clone
    numbat_gene_cna = pd.pivot_table(
        df_numbat,
        index="gene",
        columns="sample",
        values="cnv_state",
        aggfunc="first",
        fill_value="NEU",
    )

    return numbat_gene_cna


def read_starch_gene_cna(states_file):
    starch_gene_cna = pd.read_csv(states_file, header=0, index_col=0, sep=",")

    # NB STARCH output is already a gene-by-clone matrix
    starch_gene_cna = starch_gene_cna.replace({0: "DEL", 1: "NEU", 2: "AMP"})
    starch_gene_cna.columns = "clone_" + starch_gene_cna.columns + "_type"

    return starch_gene_cna


# TODO BUG null_value=0.0 -> null_value=np.nan
def compute_gene_F1(true_gene_cna, pred_gene_cna, null_value=0.0):
    """
    Compute the F1 score of CNA-affected gene prediction.

    Attributes
    ----------
    true_gene_cna : pd.DataFrame
        Each row is a gene with row index as gene name.
        Contains columns <clone>_type to indicate whether each gene in NEU, DEL, AMP, CNLOH in each clone.
    """
    F1_dict = {}

    for event in ["DEL", "AMP", "CNLOH", "overall"]:
        if event != "overall":
            # NB unique set of gene names for a given CNA type, e.g. deletion.
            # TODO BUG type definition is not relative to truth.

            # NB evalues whether any of the available clones matches the event.
            pred_event_genes = set(
                pred_gene_cna.index[np.any(pred_gene_cna == event, axis=1)]
            )
            true_event_genes = set(
                true_gene_cna.index[np.any(true_gene_cna == event, axis=1)]
            )
        else:
            pred_columns = pred_gene_cna.columns[
                pred_gene_cna.columns.str.endswith("_type")
            ]

            # TODO BUG usage of global calico_gene_cna.
            # NB all genes that are not neutral.
            pred_event_genes = set(pred_gene_cna.index) - set(
                pred_gene_cna.index[
                    np.all(pred_gene_cna[pred_columns] == "NEU", axis=1)
                ]
            )

            true_columns = true_gene_cna.columns[
                true_gene_cna.columns.str.endswith("_type")
            ]
            true_event_genes = set(true_gene_cna.index) - set(
                true_gene_cna.index[
                    np.all(true_gene_cna[true_columns] == "NEU", axis=1)
                ]
            )

        # NB some genes don't have enough coverage and are filtered out in preprocessing, so we remove them from true_event_genes.
        true_event_genes = true_event_genes & set(pred_gene_cna.index)

        if len(true_event_genes) == 0:
            F1_dict[event] = np.nan
        else:
            precision = (
                len(pred_event_genes & true_event_genes) / len(pred_event_genes)
                if len(pred_event_genes) > 0
                else null_value
            )
            recall = (
                len(pred_event_genes & true_event_genes) / len(true_event_genes)
                if len(true_event_genes) > 0
                else null_value
            )

            F1_dict[event] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else null_value
            )

    return F1_dict


def get_sim_params():
    return {
        "all_n_cnas": [(1, 2), (3, 3), (6, 3)],
        "all_cna_sizes": ["1e7", "3e7", "5e7"],
        "all_ploidy": [2],
        "all_random": np.arange(10),
    }


def get_sim_run_generator():
    sim_params = get_sim_params()

    for n_cnas in sim_params["all_n_cnas"]:
        for cna_size in sim_params["all_cna_sizes"]:
            for ploidy in sim_params["all_ploidy"]:
                for random in sim_params["all_random"]:
                    yield n_cnas, cna_size, ploidy, random


def get_sim_runs():
    run_info = []

    for n_cnas, cna_size, ploidy, random in get_sim_run_generator():
        run_info.append(
            {
                "n_cnas": n_cnas,
                "cna_size": cna_size,
                "ploidy": ploidy,
                "random": random,
                "sampleid": f"numcnas{n_cnas[0]}.{n_cnas[1]}_cnasize{cna_size}_ploidy{ploidy}_random{random}",
            }
        )

    return pd.DataFrame(run_info)


def get_sampleid(n_cnas, cna_size, ploidy, random):
    """
    Generate sampleid based on the number of CNAs - (global, shared) - CNA size, ploidy, and random seed.
    """
    return f"numcnas{n_cnas[0]}.{n_cnas[1]}_cnasize{cna_size}_ploidy{ploidy}_random{random}"


def get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random):
    sampleid = get_sampleid(n_cnas, cna_size, ploidy, random)
    return f"{true_dir}/{sampleid}/truth_clone_labels.tsv"


def get_true_clones(true_dir, n_cnas, cna_size, ploidy, random, verbose=False):
    true_clones_path = get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random)

    if verbose:
        print("Reading True clones from", true_clones_path)

    true_clones = pd.read_csv(true_clones_path, header=0, index_col=0, sep="\t")
    true_clones = true_clones.rename(columns={true_clones.columns[0]: "true_clone"})

    true_clones.index = true_clones.index.str.replace("spot_", "")
    true_clones.index.name = "spot"

    true_clones["true_clone"] = true_clones["true_clone"].str.replace("clone_", "")

    return true_clones


def __normal_clone_sorter(x):
    return (x != "normal", x)


def plot_true_clones(true_dir, n_cnas, cna_size, ploidy, random):
    true_clones = get_true_clones(true_dir, n_cnas, cna_size, ploidy, random)
    true_clones["true_clone"] = true_clones["true_clone"].str.replace("normal", "N")

    labels = true_clones["true_clone"].unique()
    labels = np.array(sorted(labels, key=__normal_clone_sorter))

    fig, ax = plt.subplots(figsize=(3, 3))

    palette = sns.color_palette()

    sns.scatterplot(
        data=true_clones,
        x="x",
        y="y",
        hue="true_clone",
        hue_order=labels,
        s=10,
        palette=palette,
    )

    handles, labels = ax.get_legend_handles_labels()
    labels = [rf"$C_{l}$" for l in labels]

    ax.set_title(
        f"{n_cnas[0]}/{n_cnas[0]} ({cna_size} MB), {random} real." + r" $\ell$",
        fontsize=10,
    )
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    return fig


def get_calico_clones_path(calico_pure_dir, n_cnas, cna_size, ploidy, random):
    sampleid = get_sampleid(n_cnas, cna_size, ploidy, random)
    r_calico = get_best_r_hmrf(get_config_path(calico_pure_dir, sampleid))

    return (
        f"{calico_pure_dir}/{sampleid}/clone3_rectangle{r_calico}_w1.0/clone_labels.tsv"
    )


def get_calico_clones(
    calico_pure_dir, n_cnas, cna_size, ploidy, random, true_dir=None, verbose=False
):
    calico_clones_path = get_calico_clones_path(
        calico_pure_dir, n_cnas, cna_size, ploidy, random
    )

    if verbose:
        print("Reading CalicoST clones from", calico_clones_path)

    calico_clones = pd.read_csv(calico_clones_path, header=0, index_col=0, sep="\t")
    calico_clones = calico_clones.rename(
        columns={calico_clones.columns[0]: "est_clone"}
    )

    calico_clones.index = calico_clones.index.str.replace("spot_", "")
    calico_clones.index.name = "spot"

    # NB join with (x,y) based on spot coordinate of truth.
    if true_dir is not None:
        true_clones = get_true_clones(true_dir, n_cnas, cna_size, ploidy, random)
        calico_clones = calico_clones.join(true_clones)

    return calico_clones


def plot_calico_clones(
    calico_pure_dir, n_cnas, cna_size, ploidy, random, true_dir, verbose=False
):
    calico_clones = get_calico_clones(
        calico_pure_dir, n_cnas, cna_size, ploidy, random, true_dir=true_dir
    )

    labels = calico_clones["est_clone"].unique()
    labels = np.array(sorted(labels, key=__normal_clone_sorter))

    fig, ax = plt.subplots(figsize=(3, 3))

    palette = sns.color_palette()

    sns.scatterplot(
        data=calico_clones,
        x="x",
        y="y",
        hue="est_clone",
        hue_order=labels,
        s=10,
        palette=palette,
    )

    handles, labels = ax.get_legend_handles_labels()
    labels = [rf"$C_{l}$" for l in labels]

    ari = adjusted_rand_score(
        calico_clones.est_clone,
        calico_clones.true_clone,
    )

    ax.set_title(
        f"{n_cnas[0]}/{n_cnas[0]} ({cna_size} MB), {random} real."
        + r" $\hat \ell$;"
        + f" ARI={ari:.2f}",
        fontsize=10,
    )

    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    return fig


def get_numbat_path(numbat_dir, sampleid):
    return f"{numbat_dir}/{sampleid}/outs/clone_post_2.tsv"


def get_numbat_clones(numbat_dir, sampleid, verbose=False):
    numbat_path = get_numbat_path(numbat_dir, sampleid)

    if Path(numbat_path).exists():
        if verbose:
            print("Reading Numbat clones from", numbat_path)

        numbat_clones = pd.read_csv(
            numbat_path,
            header=0,
            index_col=0,
            sep="\t",
        )

        numbat_clones = numbat_clones[["clone_opt"]]
        numbat_clones = numbat_clones.rename(
            columns={numbat_clones.columns[0]: "est_clone"}
        )

        numbat_clones.index = numbat_clones.index.str.replace("spot_", "")
        numbat_clones.index.name = "spot"
    else:
        numbat_clones = None

    return numbat_clones


def get_starch_path(starch_dir, sampleid):
    return f"{starch_dir}/{sampleid}/labels_STITCH_output.csv"


def get_starch_clones(starch_dir, sampleid, true_clones=None, verbose=False):
    starch_path = get_starch_path(starch_dir, sampleid)

    if verbose:
        print("Reading Starch clones from", starch_path)

    starch_clones = pd.read_csv(
        starch_path,
        header=0,
        index_col=0,
        sep=",",
        names=["est_clone"],
    )

    # NB mapper for spot index based on (x,y).
    if true_clones is not None:
        map_index = {
            f"{true_clones.x.values[i]}.0x{true_clones.y.values[i]}.0": true_clones.index[
                i
            ]
            for i in range(true_clones.shape[0])
        }

        starch_clones.index = starch_clones.index.map(map_index)

        starch_clones.index.name = "spot"

    return starch_clones


def get_base_sim_summary(n_cnas, cna_size, ploidy, random, sampleid, true_path):
    map_cnasize = {"1e7": "10Mb", "3e7": "30Mb", "5e7": "50Mb"}

    return pd.DataFrame(
        {
            "cnas": f"{n_cnas[0], n_cnas[1]}",
            "n_cnas": n_cnas[0] + n_cnas[1],
            "cna_size": map_cnasize[cna_size],
            "ploidy": int(ploidy),
            "random": random,
            "sample_id": sampleid,
            "true_clones_path": true_path,
        },
        index=[0],
    )


def get_pair_recall(est_label, true_label):
    # NB not distinct pairs,
    #    see https://scikit-learn.org/dev/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html
    confusion = pair_confusion_matrix(true_label, est_label)
    _, cnts = np.unique(true_label, return_counts=True)

    return confusion[1, 1] / (cnts * (cnts - 1)).sum()


def get_aris(true_dir, calico_pure_dir, numbat_dir, starch_dir):
    sim_params = get_sim_params()
    df_clone_ari = []

    for n_cnas, cna_size, ploidy, random in get_sim_run_generator():
        true_path = get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random)
        true_clones = get_true_clones(true_dir, n_cnas, cna_size, ploidy, random)

        # sampleid = f"numcnas{n_cnas[0]}.{n_cnas[1]}_cnasize{cna_size}_ploidy{ploidy}_random{random}"
        sampleid = get_sampleid(n_cnas, cna_size, ploidy, random)

        # CalicoST
        best_fit_clones_path = get_calico_clones_path(
            calico_pure_dir, n_cnas, cna_size, ploidy, random
        )

        calico_pure_clones = get_calico_clones(
            calico_pure_dir, n_cnas, cna_size, ploidy, random
        )
        calico_pure_clones = calico_pure_clones.join(true_clones)

        base_summary = get_base_sim_summary(
            n_cnas, cna_size, ploidy, random, sampleid, true_path
        )

        # TODO "r_calico": r_calico,
        calicost_summary = base_summary.copy()
        calicost_summary["method"] = "CalicoST"

        # NB fraction of all spot pairs that share a clone in truth, and share a clone in estimation.
        calicost_summary["recall"] = get_pair_recall(
            calico_pure_clones.est_clone, calico_pure_clones.true_clone
        )

        calicost_summary["ari"] = adjusted_rand_score(
            calico_pure_clones.est_clone,
            calico_pure_clones.true_clone,
        )
        calicost_summary["best_fit_clones_path"] = best_fit_clones_path

        df_clone_ari.append(calicost_summary)

        # Numbat
        numbat_path = get_numbat_path(numbat_dir, sampleid)
        numbat_clones = get_numbat_clones(numbat_dir, sampleid)

        numbat_summary = base_summary.copy()
        numbat_summary["method"] = "Numbat"
        numbat_summary["best_fit_clones_path"] = numbat_path

        if numbat_clones is not None:
            numbat_clones = numbat_clones.join(true_clones)
            numbat_summary["ari"] = adjusted_rand_score(
                numbat_clones.est_clone,
                numbat_clones.true_clone,
            )

        else:
            numbat_summary["ari"] = 0.0
            numbat_summary["best_fit_clones_path"] = "-"

        df_clone_ari.append(numbat_summary)

        # STARCH
        starch_path = get_starch_path(starch_dir, sampleid)

        starch_clones = get_starch_clones(starch_dir, sampleid, true_clones)

        starch_clones = starch_clones.join(true_clones)

        starch_summary = base_summary.copy()
        starch_summary["method"] = "STARCH"
        starch_summary["ari"] = adjusted_rand_score(
            starch_clones.est_clone,
            starch_clones.true_clone,
        )
        starch_summary["best_fit_clones_path"] = starch_path

        df_clone_ari.append(starch_summary)

    # TODO .sort_values(by='cna_size', ascending=False)
    df_clone_ari = pd.concat(df_clone_ari, ignore_index=True)
    df_clone_ari.cna_size = pd.Categorical(
        df_clone_ari.cna_size, categories=["10Mb", "30Mb", "50Mb"], ordered=True
    )

    # NB scrollable columns.
    df_clone_ari.style.set_table_styles(
        [{"selector": "th", "props": [("position", "sticky"), ("top", "0")]}],
        overwrite=False,
    ).set_sticky(axis="columns")

    return df_clone_ari


def plot_aris(df_clone_ari):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, dpi=300)
    color_methods = ["#4963C1", "#97b085", "#dad67d"]

    palette = sns.color_palette()

    for i, n_cnas in enumerate([3, 6, 9]):
        tmpdf = df_clone_ari[df_clone_ari.n_cnas == n_cnas]

        sns.boxplot(
            data=tmpdf,
            x="cna_size",
            y="ari",
            hue="method",
            palette=palette,
            boxprops=dict(alpha=0.7),
            linewidth=1,
            showfliers=False,
            ax=axes[i],
        )

        sns.stripplot(
            data=tmpdf,
            x="cna_size",
            y="ari",
            hue="method",
            palette=palette,
            dodge=10,
            edgecolor="black",
            linewidth=0.5,
            ax=axes[i],
        )

        if (i + 1) < 3:
            axes[i].get_legend().remove()

        axes[i].set_ylim(0.0, 1.0)
        axes[i].set_ylabel("Clone ARI")
        axes[i].set_xlabel("Simulated CNA length")
        axes[i].set_title(f"{n_cnas} CNA events")

    h, l = axes[-1].get_legend_handles_labels()
    axes[-1].legend(h[:3], l[:3], loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.show()


def get_truth_cna_file(true_dir, sampleid):
    return f"{true_dir}/{sampleid}/truth_cna.tsv"


def get_numbat_cna_file(numbat_dir, sampleid):
    return f"{numbat_dir}/{sampleid}/outs/bulk_clones_final.tsv.gz"


def get_starch_cna_file(starch_dir, sampleid):
    return f"{starch_dir}/{sampleid}/states_STITCH_output.csv"


def get_f1s(true_dir, df_hgtable, calico_pure_dir, numbat_dir, starch_dir):
    # EG 6 shared CNAs and 3 clone specific.
    sim_params = get_sim_params()
    list_events = ["DEL", "AMP", "CNLOH", "overall"]

    df_event_f1 = []

    for n_cnas, cna_size, ploidy, random in get_sim_run_generator():
        sampleid = get_sampleid(n_cnas, cna_size, ploidy, random)

        truth_cna_file = get_truth_cna_file(true_dir, sampleid)
        true_gene_cna = read_true_gene_cna(df_hgtable, truth_cna_file)

        base_summary = get_base_sim_summary(
            n_cnas, cna_size, ploidy, random, sampleid, truth_cna_file
        )

        # CalicoST
        configuration_file = get_config_path(calico_pure_dir, sampleid)
        calico_gene_cna = read_calico_gene_cna(configuration_file)

        F1_dict = compute_gene_F1(true_gene_cna, calico_gene_cna)

        calicost_summary = base_summary.copy()
        calicost_summary["method"] = "CalicoST"
        calicost_summary["event"] = [list_events]
        calicost_summary["F1"] = [[F1_dict[e] for e in list_events]]
        calicost_summary["true_cna"] = truth_cna_file
        calicost_summary["est_cna_file"] = get_calico_cna_file(configuration_file)

        df_event_f1.append(
            calicost_summary.explode(["event", "F1"]).reset_index(drop=True)
        )

        # Numbat
        numbat_cna_file = get_numbat_cna_file(numbat_dir, sampleid)

        numbat_summary = base_summary.copy()
        numbat_summary["method"] = "Numbat"
        numbat_summary["event"] = [list_events]
        numbat_summary["true_cna"] = truth_cna_file

        if Path(numbat_cna_file).exists():
            numbat_gene_cna = read_numbat_gene_cna(numbat_cna_file)

            F1_dict = compute_gene_F1(true_gene_cna, numbat_gene_cna)

            numbat_summary["F1"] = [[F1_dict[e] for e in list_events]]
            numbat_summary["est_cna_file"] = numbat_cna_file
        else:
            numbat_summary["F1"] = [[0.0 for e in list_events]]
            numbat_summary["est_cna_file"] = "-"

        df_event_f1.append(
            numbat_summary.explode(["event", "F1"]).reset_index(drop=True)
        )

        # STARCH
        starch_cna_file = get_starch_cna_file(starch_dir, sampleid)
        starch_gene_cna = read_starch_gene_cna(starch_cna_file)

        F1_dict = compute_gene_F1(true_gene_cna, starch_gene_cna)

        starch_summary = base_summary.copy()
        starch_summary["method"] = "STARCH"
        starch_summary["event"] = [list_events]
        starch_summary["F1"] = [[F1_dict[e] for e in list_events]]
        starch_summary["true_cna"] = truth_cna_file
        starch_summary["est_cna_file"] = starch_cna_file

        df_event_f1.append(
            starch_summary.explode(["event", "F1"]).reset_index(drop=True)
        )

    return pd.concat(df_event_f1, ignore_index=True)


def plot_f1s(df_event_f1):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    color_methods = ["#4963C1", "#97b085", "#dad67d"]

    for i, n_cnas in enumerate([3, 6, 9]):
        tmpdf = df_event_f1[df_event_f1.n_cnas == n_cnas]

        sns.boxplot(
            data=tmpdf,
            x="event",
            y="F1",
            hue="method",
            palette=sns.color_palette(color_methods),
            boxprops=dict(alpha=0.7),
            linewidth=1,
            showfliers=False,
            ax=axes[i],
        )

        sns.stripplot(
            data=tmpdf,
            x="event",
            y="F1",
            hue="method",
            palette=sns.color_palette(color_methods),
            dodge=10,
            edgecolor="black",
            linewidth=0.5,
            ax=axes[i],
        )

        if i + 1 < 3:
            axes[i].get_legend().remove()

        axes[i].set_ylabel("F1")
        axes[i].set_xlabel(None)
        axes[i].set_title(f"{n_cnas} CNA events")
        axes[i].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    h, l = axes[-1].get_legend_handles_labels()
    axes[-1].legend(h[:3], l[:3], loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.show()

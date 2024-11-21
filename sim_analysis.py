from pathlib import Path

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix

from calicost import arg_parse
from calicost.utils_plotting import get_full_palette
from calicost.utils_hmrf import merge_pseudobulk_by_index, merge_pseudobulk_by_index_mix
from calicost.utils_phase_switch import get_intervals


__cnasize_mapper = {"1e7": "10Mb", "3e7": "30Mb", "5e7": "50Mb"}


def path_not_exists(path):
    if not Path(path).exists():
        warnings.warn(f"\n{path} does not exist.")
        return True

    return False


def get_gene_ranges_path(calico_repo_dir):
    return f"{calico_repo_dir}/GRCh38_resources/hgTables_hg38_gencode.txt"


def read_gene_ranges(calico_repo_dir):
    """
    Read in (chronological order, start, end) table for (coding) gene definition.
    """
    gene_ranges_path = get_gene_ranges_path(calico_repo_dir)

    gene_ranges = pd.read_csv(gene_ranges_path, sep="\t", header=0, index_col=0)
    gene_ranges = gene_ranges[gene_ranges.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    # NB add chr column as integer without "chr" prefix
    gene_ranges["chr"] = [int(x[3:]) for x in gene_ranges.chrom]
    gene_ranges = gene_ranges.rename(
        columns={"cdsStart": "start", "cdsEnd": "end", "name2": "gene"}
    )
    gene_ranges.set_index("gene", inplace=True)

    return gene_ranges[["chr", "start", "end"]]


def get_simid(n_cnas, cna_size, ploidy, random):
    """
    Generate simid based on the number of CNAs - (global, shared) - CNA size, ploidy, and random seed.
    """
    return f"numcnas{n_cnas[0]}.{n_cnas[1]}_cnasize{cna_size}_ploidy{ploidy}_random{random}"


def get_config_path(calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    return f"{calico_dir}/{simid}/configfile{initialization_seed}"


def get_config(
    calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed, verbose=False
):
    """
    Retrieve the CalicoST config.
    """
    configuration_file = get_config_path(
        calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
    )
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
            print(f"  {key}: {config[key]}")

    return config


def get_calico_realization_results_path(
    calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    config = get_config(
        calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
    )

    if config is not None:
        return f"{calico_dir}/{simid}/clone{config['n_clones']}_rectangle{initialization_seed}_w{config['spatial_weight']:.1f}"
    else:
        return None


def get_rdrbaf_path(calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed):
    config = get_config(
        calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
    )

    if config is not None:
        results_dir_path = get_calico_realization_results_path(
            calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
        )

        return f"{results_dir_path}/rdrbaf_final_nstates{config['n_states']}_smp.npz"
    else:
        return None


def get_rdrbaf(
    calico_dir,
    n_cnas,
    cna_size,
    ploidy,
    random,
    initialization_seed,
    verbose=False,
):
    """
    Retrieve the CalicoST RDR/BAF determinations.
    """
    config = get_config(
        calico_dir,
        n_cnas,
        cna_size,
        ploidy,
        random,
        initialization_seed,
        verbose=verbose,
    )
    rdrbaf_path = get_rdrbaf_path(
        calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
    )

    if rdrbaf_path is not None and Path(rdrbaf_path).exists():
        return dict(
            np.load(rdrbaf_path),
            allow_pickle=True,
        )
    else:
        if verbose:
            warnings.warn(
                f"CalicoST RDR/BAF determinations do not exist for {rdrbaf_path}"
            )
        return None


def get_r_hmrf_likelihoods(
    calico_dir,
    n_cnas,
    cna_size,
    ploidy,
    random,
    verbose=False,
):
    """
    Retrieve the CalicoST random initializations likelihoods.
    """
    config = get_config(
        calico_dir,
        n_cnas,
        cna_size,
        ploidy,
        random,
        initialization_seed=0,
        verbose=verbose,
    )

    # NB find the best HMRF initialization random seed
    df_clone = []

    for initialization_seed in range(10):
        rdrbaf = get_rdrbaf(
            calico_dir,
            n_cnas,
            cna_size,
            ploidy,
            random,
            initialization_seed,
        )

        if rdrbaf is not None:
            df_clone.append(
                pd.DataFrame(
                    {
                        "initialization_seed": initialization_seed,
                        "log_likelihood": rdrbaf["total_llf"],
                    },
                    index=[0],
                )
            )
        else:
            rdrbaf_path = get_rdrbaf_path(
                calico_dir, n_cnas, cna_size, ploidy, random, initialization_seed
            )
            warnings.warn(f"{rdrbaf_path} does not exist.")

    return pd.concat(df_clone, ignore_index=True) if len(df_clone) > 0 else None


def get_best_r_hmrf(
    calico_dir,
    n_cnas,
    cna_size,
    ploidy,
    random,
    verbose=False,
):
    """
    Retrieve the CalicoST random initialization with the maximum likelihood.
    """
    df_clone = get_r_hmrf_likelihoods(calico_dir, n_cnas, cna_size, ploidy, random)

    # NB returns first of degenerate max., i.e. 0 if all the likelihoods are the same.
    return (
        int(df_clone["random_seed"].iloc[np.argmax(df_clone["log_likelihood"])])
        if df_clone is not None
        else -1
    )


# TODO relation to cnv_genelevel.tsv?
def get_cna_seglevel_path(calico_dir, simid, r_hmrf_initialization, ploidy="diploid"):
    # TODO assumes clone3.
    # e.g. ../nomixing_calicost_related/numcnas1.2_cnasize1e7_ploidy2_random0/clone3_rectangle0_w1.0/cnv_diploid_seglevel.tsv
    return f"{calico_dir}/{simid}/clone3_rectangle{r_hmrf_initialization}_w1.0/cnv_{ploidy}_seglevel.tsv"


def filter_non_netural(cna_frame):
    """
    Filter out neutral CNAs.
    """
    columns = cna_frame.columns

    isin = columns.str.contains("clone")
    isin &= [not xx for xx in columns.str.contains("type")]

    clones = columns[isin].values

    isin, isglobal = [], []

    for index, row in cna_frame.iterrows():
        isin.append(not np.all(row[clones] == 1))
        isglobal.append(row.nunique() == 1)

    cna_seglevel = cna_frame[isin].reset_index(drop=True)
    cna_seglevel["global_cna"] = np.array(isglobal)[isin]

    return cna_seglevel


def get_cna_seglevel(
    calico_dir,
    simid,
    r_hmrf_initialization,
    ploidy="diploid",
    non_neutral_only=False,
):
    cna_seglevel_path = get_cna_seglevel_path(
        calico_dir, simid, r_hmrf_initialization, ploidy=ploidy
    )

    cna_seglevel = pd.read_csv(cna_seglevel_path, header=0, sep="\t")
    cna_seglevel.columns = [xx.lower() for xx in cna_seglevel.columns]

    if non_neutral_only:
        cna_seglevel = filter_non_netural(cna_seglevel)

    return cna_seglevel


def plot_rdr_baf(
    calico_dir,
    n_cnas,
    cna_size,
    ploidy,
    random,
    clone_ids=None,
    clone_names=None,
    remove_xticks=True,
    rdr_ylim=5,
    chrtext_shift=-0.3,
    base_height=3.2,
    pointsize=15,
    linewidth=1,
    palette="chisel",
):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    configuration_file = get_config_path(calico_dir, n_cnas, cna_size, ploidy, random)

    r_hmrf_initialization = get_best_r_hmrf(
        calico_dir, n_cnas, cna_size, ploidy, random
    )

    cna_path = get_cna_seglevel_path(calico_dir, simid, r_hmrf_initialization)
    df_cnv = get_cna_seglevel(calico_dir, simid, r_hmrf_initialization)
    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])

    chisel_palette, ordered_acn = get_full_palette()
    map_cn = {x: i for i, x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    config = get_config(calico_dir, n_cnas, cna_size, ploidy, random)

    if not "0" in final_clone_ids:
        final_clone_ids = np.array(["0"] + list(final_clone_ids))

    assert (clone_ids is None) or np.all(
        [(cid in final_clone_ids) for cid in clone_ids]
    )

    unique_chrs = np.unique(df_cnv.chr.values)

    outdir = f"{calico_dir}/{simid}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"

    dat = np.load(f"{outdir}/binned_data.npz", allow_pickle=True)
    lengths = dat["lengths"]
    single_X = dat["single_X"]
    single_base_nb_mean = dat["single_base_nb_mean"]
    single_total_bb_RD = dat["single_total_bb_RD"]
    single_tumor_prop = dat["single_tumor_prop"]
    res_combine = dict(
        np.load(
            f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz",
            allow_pickle=True,
        )
    )

    n_states = res_combine["new_p_binom"].shape[0]

    assert (
        single_X.shape[0] == df_cnv.shape[0]
    ), f"Found single_X.shape[0] == {single_X.shape[0]} but expected df_cnv.shape[0] == {df_cnv.shape[0]}"

    clone_index = [
        np.where(res_combine["new_assignment"] == c)[0]
        for c, cid in enumerate(final_clone_ids)
    ]

    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(
            single_X, single_base_nb_mean, single_total_bb_RD, clone_index
        )
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
        )

    n_obs = X.shape[0]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    if clone_ids is None:
        fig, axes = plt.subplots(
            2 * len(nonempty_clones),
            1,
            figsize=(20, base_height * len(nonempty_clones)),
            dpi=200,
            facecolor="white",
        )

        for s, c in enumerate(nonempty_clones):
            cid = final_clone_ids[c]

            # NB major and minor allele copies give the hue
            major = np.maximum(
                df_cnv[f"clone{cid} a"].values, df_cnv[f"clone{cid} b"].values
            )
            minor = np.minimum(
                df_cnv[f"clone{cid} a"].values, df_cnv[f"clone{cid} b"].values
            )

            segments, labs = get_intervals(res_combine["pred_cnv"][:, c])

            if palette == "chisel":
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 0, c] / base_nb_mean[:, c],
                    hue=pd.Categorical(
                        [map_cn[(major[i], minor[i])] for i in range(len(major))],
                        categories=np.arange(len(ordered_acn)),
                        ordered=True,
                    ),
                    palette=sns.color_palette(colors),
                    s=pointsize,
                    edgecolor="black",
                    linewidth=linewidth,
                    alpha=1,
                    legend=False,
                    ax=axes[2 * s],
                )
            else:
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 0, c] / base_nb_mean[:, c],
                    hue=pd.Categorical(
                        res_combine["pred_cnv"][:, c],
                        categories=np.arange(n_states),
                        ordered=True,
                    ),
                    palette=palette,
                    s=pointsize,
                    edgecolor="black",
                    linewidth=linewidth,
                    alpha=1,
                    legend=False,
                    ax=axes[2 * s],
                )

            axes[2 * s].set_ylabel(f"clone {cid}\nRDR")
            axes[2 * s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2 * s].set_ylim([0, rdr_ylim])
            axes[2 * s].set_xlim([0, n_obs])

            if remove_xticks:
                axes[2 * s].set_xticks([])

            if palette == "chisel":
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 1, c] / total_bb_RD[:, c],
                    hue=pd.Categorical(
                        [map_cn[(major[i], minor[i])] for i in range(len(major))],
                        categories=np.arange(len(ordered_acn)),
                        ordered=True,
                    ),
                    palette=sns.color_palette(colors),
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s + 1],
                )
            else:
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 1, c] / total_bb_RD[:, c],
                    hue=pd.Categorical(
                        res_combine["pred_cnv"][:, c],
                        categories=np.arange(n_states),
                        ordered=True,
                    ),
                    palette=palette,
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s + 1],
                )

            axes[2 * s + 1].set_ylabel(f"clone {cid}\nphased AF")
            axes[2 * s + 1].set_ylim([-0.1, 1.1])
            axes[2 * s + 1].set_yticks([0, 0.5, 1])
            axes[2 * s + 1].set_xlim([0, n_obs])

            if remove_xticks:
                axes[2 * s + 1].set_xticks([])

            for i, seg in enumerate(segments):
                axes[2 * s].plot(
                    seg,
                    [
                        np.exp(res_combine["new_log_mu"][labs[i], c]),
                        np.exp(res_combine["new_log_mu"][labs[i], c]),
                    ],
                    c="black",
                    linewidth=2,
                )
                axes[2 * s + 1].plot(
                    seg,
                    [
                        res_combine["new_p_binom"][labs[i], c],
                        res_combine["new_p_binom"][labs[i], c],
                    ],
                    c="black",
                    linewidth=2,
                )
                axes[2 * s + 1].plot(
                    seg,
                    [
                        1 - res_combine["new_p_binom"][labs[i], c],
                        1 - res_combine["new_p_binom"][labs[i], c],
                    ],
                    c="black",
                    linewidth=2,
                )

        for i in range(len(lengths)):
            median_len = (
                np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[: (i + 1)]) * 0.45
            )
            axes[-1].text(
                median_len - 5,
                chrtext_shift,
                unique_chrs[i],
                transform=axes[-1].get_xaxis_transform(),
            )
            for k in range(2 * len(nonempty_clones)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()

    else:
        fig, axes = plt.subplots(
            2 * len(clone_ids),
            1,
            figsize=(20, base_height * len(clone_ids)),
            dpi=200,
            facecolor="white",
        )
        for s, cid in enumerate(clone_ids):
            c = np.where(final_clone_ids == cid)[0][0]

            # major and minor allele copies give the hue
            major = np.maximum(
                df_cnv[f"clone{cid} a"].values, df_cnv[f"clone{cid} b"].values
            )
            minor = np.minimum(
                df_cnv[f"clone{cid} a"].values, df_cnv[f"clone{cid} b"].values
            )

            segments, labs = get_intervals(res_combine["pred_cnv"][:, c])
            if palette == "chisel":
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 0, c] / base_nb_mean[:, c],
                    hue=pd.Categorical(
                        [map_cn[(major[i], minor[i])] for i in range(len(major))],
                        categories=np.arange(len(ordered_acn)),
                        ordered=True,
                    ),
                    palette=sns.color_palette(colors),
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s],
                )
            else:
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 0, c] / base_nb_mean[:, c],
                    hue=pd.Categorical(
                        res_combine["pred_cnv"][:, c],
                        categories=np.arange(n_states),
                        ordered=True,
                    ),
                    palette=palette,
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s],
                )
            axes[2 * s].set_ylabel(
                f"clone {cid}\nRDR"
                if clone_names is None
                else f"clone {clone_names[s]}\nRDR"
            )
            axes[2 * s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2 * s].set_ylim([0, 5])
            axes[2 * s].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2 * s].set_xticks([])
            if palette == "chisel":
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 1, c] / total_bb_RD[:, c],
                    hue=pd.Categorical(
                        [map_cn[(major[i], minor[i])] for i in range(len(major))],
                        categories=np.arange(len(ordered_acn)),
                        ordered=True,
                    ),
                    palette=sns.color_palette(colors),
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s + 1],
                )
            else:
                sns.scatterplot(
                    x=np.arange(X[:, 1, c].shape[0]),
                    y=X[:, 1, c] / total_bb_RD[:, c],
                    hue=pd.Categorical(
                        res_combine["pred_cnv"][:, c],
                        categories=np.arange(n_states),
                        ordered=True,
                    ),
                    palette=palette,
                    s=pointsize,
                    edgecolor="black",
                    alpha=0.8,
                    legend=False,
                    ax=axes[2 * s + 1],
                )
            axes[2 * s + 1].set_ylabel(
                f"clone {cid}\nphased AF"
                if clone_names is None
                else f"clone {clone_names[s]}\nphased AF"
            )
            axes[2 * s + 1].set_ylim([-0.1, 1.1])
            axes[2 * s + 1].set_yticks([0, 0.5, 1])
            axes[2 * s + 1].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2 * s + 1].set_xticks([])
            for i, seg in enumerate(segments):
                axes[2 * s].plot(
                    seg,
                    [
                        np.exp(res_combine["new_log_mu"][labs[i], c]),
                        np.exp(res_combine["new_log_mu"][labs[i], c]),
                    ],
                    c="black",
                    linewidth=2,
                )
                axes[2 * s + 1].plot(
                    seg,
                    [
                        res_combine["new_p_binom"][labs[i], c],
                        res_combine["new_p_binom"][labs[i], c],
                    ],
                    c="black",
                    linewidth=2,
                )
                axes[2 * s + 1].plot(
                    seg,
                    [
                        1 - res_combine["new_p_binom"][labs[i], c],
                        1 - res_combine["new_p_binom"][labs[i], c],
                    ],
                    c="black",
                    linewidth=2,
                )

        for i in range(len(lengths)):
            median_len = (
                np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[: (i + 1)]) * 0.45
            )
            axes[-1].text(
                median_len - 5,
                chrtext_shift,
                unique_chrs[i],
                transform=axes[-1].get_xaxis_transform(),
            )
            for k in range(2 * len(clone_ids)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()

    return fig


def read_true_gene_cna(gene_ranges, truth_cna_file, non_neutral_only=False):
    """
    Read true copy number aberrations
    """
    true_gene_cna = gene_ranges.copy()
    df_cna = pd.read_csv(truth_cna_file, header=0, index_col=0, sep="\t")

    unique_clones = df_cna.index.unique()

    for clonename in unique_clones:
        clonal_cna = df_cna[df_cna.index == clonename]

        gene_A = np.array([1] * true_gene_cna.shape[0])
        gene_B = np.array([1] * true_gene_cna.shape[0])

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

            gene_A[is_affected] = clonal_cna.A_copy.values[i]
            gene_B[is_affected] = clonal_cna.B_copy.values[i]

            gene_status_str[is_affected] = cna_str
            gene_status_type[is_affected] = cna_type

        clonename = clonename.replace("_", "")

        true_gene_cna[f"{clonename} a"] = gene_A
        true_gene_cna[f"{clonename} b"] = gene_B

        true_gene_cna[f"{clonename}_gtype"] = gene_status_str
        true_gene_cna[f"{clonename}_ctype"] = gene_status_type

    if non_neutral_only:
        true_gene_cna = filter_non_netural(true_gene_cna)

    return true_gene_cna


def get_calico_cna_file(calico_dir, n_cnas, cna_size, ploidy, random):
    config_path = get_config_path(calico_dir, n_cnas, cna_size, ploidy, random)
    r_calico = get_best_r_hmrf(calico_dir, n_cnas, cna_size, ploidy, random)

    return (
        f"{Path(config_path).parent}/clone3_rectangle{r_calico}_w1.0/cnv_genelevel.tsv"
    )


def read_calico_gene_cna(calico_dir, n_cnas, cna_size, ploidy, random):
    """
    Read CalicoST estimated copy number aberrations.
    """
    fpath = get_calico_cna_file(calico_dir, n_cnas, cna_size, ploidy, random)

    if path_not_exists(fpath):
        return None

    calico_gene_cna = pd.read_csv(
        fpath,
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

    for event in ["DEL", "AMP", "CNLOH", "ALL"]:
        if event != "ALL":
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


def get_sim_runs(calico_dir=None):
    df = pd.DataFrame(
        [
            {
                "n_cnas": n_cnas,
                "cna_size": cna_size,
                "ploidy": ploidy,
                "random": random,
                "simid": get_simid(n_cnas, cna_size, ploidy, random),
            }
            for n_cnas, cna_size, ploidy, random in get_sim_run_generator()
        ]
    )

    # NB 3x (global, local), 3x CNA size, 10x random.
    assert len(df) == 90

    if calico_dir is not None:
        likelihoods = [
            get_r_hmrf_likelihoods(calico_dir, n_cnas, cna_size, ploidy, random)
            for n_cnas, cna_size, ploidy, random in get_sim_run_generator()
        ]
        df["nvalid"] = list(
            map(lambda xx: len(xx) if xx is not None else 0, likelihoods)
        )

    return df


def get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    return f"{true_dir}/{simid}/truth_clone_labels.tsv"


def get_true_clones(true_dir, n_cnas, cna_size, ploidy, random, verbose=False):
    true_clones_path = get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random)

    if verbose:
        print("Reading True clones from", true_clones_path)

    true_clones = pd.read_csv(true_clones_path, header=0, index_col=0, sep="\t")
    true_clones = true_clones.rename(columns={true_clones.columns[0]: "true_clone"})

    true_clones.index = true_clones.index.str.replace("spot_", "")
    true_clones.index.name = "spot"

    true_clones["true_clone"] = (
        true_clones["true_clone"].str.replace("clone_", "").str.replace("normal", "N")
    )

    return true_clones


def __normal_clone_sorter(x):
    return (x != "normal", x)


def plot_clones(clones, n_cnas, cna_size, ploidy, random, truth=False):
    if clones is None:
        warnings.warn("No clones to plot.")
        return

    clones = clones.copy()

    column = "true_clone" if truth else "est_clone"

    labels = clones[column].unique()
    labels = np.array(sorted(labels, key=__normal_clone_sorter))

    title = f"{n_cnas[0]}/{n_cnas[0]} ({cna_size} MB), {random} real." + r" $\ell$"

    if not truth:
        ari = adjusted_rand_score(
            clones.est_clone,
            clones.true_clone,
        )

        title += f" ARI={ari:.2f}"

    fig, ax = plt.subplots(figsize=(3, 3))

    palette = sns.color_palette()

    sns.scatterplot(
        data=clones,
        x="x",
        y="y",
        hue=column,
        hue_order=labels,
        s=10,
        palette=palette,
    )

    handles, labels = ax.get_legend_handles_labels()
    labels = [rf"$C_{l}$" for l in labels]

    ax.set_title(
        title,
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


def get_calico_best_clones_path(calico_dir, n_cnas, cna_size, ploidy, random):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    r_calico = get_best_r_hmrf(
        calico_dir,
        n_cnas,
        cna_size,
        ploidy,
        random,
    )

    # TODO results dir path.
    return f"{calico_dir}/{simid}/clone3_rectangle{r_calico}_w1.0/clone_labels.tsv"


def get_calico_best_clones(
    calico_dir, n_cnas, cna_size, ploidy, random, true_dir=None, verbose=False
):
    calico_clones_path = get_calico_best_clones_path(
        calico_dir, n_cnas, cna_size, ploidy, random
    )

    if path_not_exists(calico_clones_path):
        return None

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


def get_numbat_path(numbat_dir, n_cnas, cna_size, ploidy, random):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    return f"{numbat_dir}/{simid}/outs/clone_post_2.tsv"


def get_numbat_clones(
    numbat_dir, n_cnas, cna_size, ploidy, random, true_clones=None, verbose=False
):
    numbat_path = get_numbat_path(numbat_dir, n_cnas, cna_size, ploidy, random)

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

        if true_clones is not None:
            numbat_clones = numbat_clones.join(true_clones)

    else:
        warnings.warn(f"\n{numbat_path} does not exist.")
        numbat_clones = None

    return numbat_clones


def get_starch_path(starch_dir, n_cnas, cna_size, ploidy, random):
    simid = get_simid(n_cnas, cna_size, ploidy, random)
    return f"{starch_dir}/{simid}/labels_STITCH_output.csv"


def get_starch_clones(
    starch_dir, n_cnas, cna_size, ploidy, random, true_clones=None, verbose=False
):
    starch_path = get_starch_path(starch_dir, n_cnas, cna_size, ploidy, random)

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

        starch_clones = starch_clones.join(true_clones)

    return starch_clones


def get_base_sim_summary(n_cnas, cna_size, ploidy, random, simid, true_path):
    return pd.DataFrame(
        {
            "cnas": f"{n_cnas[0], n_cnas[1]}",
            "n_cnas": n_cnas[0] + n_cnas[1],
            "cna_size": __cnasize_mapper[cna_size],
            "ploidy": int(ploidy),
            "random": random,
            "sample_id": simid,
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


def get_clone_aris(true_dir, calico_dir, numbat_dir, starch_dir):
    sim_params = get_sim_params()
    df_clone_ari = []

    for n_cnas, cna_size, ploidy, random in get_sim_run_generator():
        true_path = get_true_clones_path(true_dir, n_cnas, cna_size, ploidy, random)
        true_clones = get_true_clones(true_dir, n_cnas, cna_size, ploidy, random)

        simid = get_simid(n_cnas, cna_size, ploidy, random)

        base_summary = get_base_sim_summary(
            n_cnas, cna_size, ploidy, random, simid, true_path
        )

        # -- CalicoST --
        best_clones_path = get_calico_best_clones_path(
            calico_dir, n_cnas, cna_size, ploidy, random
        )

        calico_clones = get_calico_best_clones(
            calico_dir, n_cnas, cna_size, ploidy, random
        )

        if calico_clones is not None:
            calico_clones = calico_clones.join(true_clones)

            calicost_summary = base_summary.copy()
            calicost_summary["method"] = "CalicoST"

            # NB fraction of all spot pairs that share a clone in truth, and share a clone in estimation.
            calicost_summary["recall"] = get_pair_recall(
                calico_clones.est_clone, calico_clones.true_clone
            )

            calicost_summary["ari"] = adjusted_rand_score(
                calico_clones.est_clone,
                calico_clones.true_clone,
            )
            calicost_summary["best_clones_path"] = best_clones_path

            df_clone_ari.append(calicost_summary)

        # -- Numbat --
        numbat_path = get_numbat_path(numbat_dir, n_cnas, cna_size, ploidy, random)
        numbat_clones = get_numbat_clones(numbat_dir, n_cnas, cna_size, ploidy, random)

        numbat_summary = base_summary.copy()
        numbat_summary["method"] = "Numbat"
        numbat_summary["best_clones_path"] = numbat_path

        if numbat_clones is not None:
            numbat_clones = numbat_clones.join(true_clones)
            numbat_summary["recall"] = get_pair_recall(
                numbat_clones.est_clone, numbat_clones.true_clone
            )
            numbat_summary["ari"] = adjusted_rand_score(
                numbat_clones.est_clone,
                numbat_clones.true_clone,
            )

        else:
            numbat_summary["recall"] = 0.0
            numbat_summary["ari"] = 0.0
            numbat_summary["best_clones_path"] = numbat_path

        df_clone_ari.append(numbat_summary)

        # -- STARCH --
        starch_path = get_starch_path(starch_dir, n_cnas, cna_size, ploidy, random)

        starch_clones = get_starch_clones(
            starch_dir,
            n_cnas,
            cna_size,
            ploidy,
            random,
            true_clones=true_clones,
        )

        starch_summary = base_summary.copy()
        starch_summary["method"] = "Starch"
        starch_summary["recall"] = get_pair_recall(
            starch_clones.est_clone, starch_clones.true_clone
        )
        starch_summary["ari"] = adjusted_rand_score(
            starch_clones.est_clone,
            starch_clones.true_clone,
        )
        starch_summary["best_clones_path"] = starch_path

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


def plot_clone_aris(df_clone_ari):
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
    axes[-1].legend(
        h[:3], l[:3], loc="upper left", bbox_to_anchor=(1, 1), frameon=False
    )

    fig.tight_layout()
    fig.show()


def get_truth_cna_file(true_dir, simid):
    return f"{true_dir}/{simid}/truth_cna.tsv"


def get_numbat_cna_file(numbat_dir, simid):
    return f"{numbat_dir}/{simid}/outs/bulk_clones_final.tsv.gz"


def get_starch_cna_file(starch_dir, simid):
    return f"{starch_dir}/{simid}/states_STITCH_output.csv"


def get_cna_f1s(calico_repo_dir, true_dir, calico_dir, numbat_dir, starch_dir):
    gene_ranges = read_gene_ranges(calico_repo_dir)

    # EG 6 shared CNAs and 3 clone specific.
    sim_params = get_sim_params()
    list_events = ["DEL", "AMP", "CNLOH", "ALL"]

    df_event_f1 = []

    for n_cnas, cna_size, ploidy, random in get_sim_run_generator():
        simid = get_simid(n_cnas, cna_size, ploidy, random)

        truth_cna_file = get_truth_cna_file(true_dir, simid)
        true_gene_cna = read_true_gene_cna(gene_ranges, truth_cna_file)

        base_summary = get_base_sim_summary(
            n_cnas, cna_size, ploidy, random, simid, truth_cna_file
        )

        # CalicoST
        configuration_file = get_config_path(
            calico_dir, n_cnas, cna_size, ploidy, random
        )
        calico_gene_cna = read_calico_gene_cna(
            calico_dir, n_cnas, cna_size, ploidy, random
        )

        if calico_gene_cna is not None:
            F1_dict = compute_gene_F1(true_gene_cna, calico_gene_cna)

            calicost_summary = base_summary.copy()
            calicost_summary["method"] = "CalicoST"
            calicost_summary["event"] = [list_events]
            calicost_summary["F1"] = [[F1_dict[e] for e in list_events]]
            calicost_summary["true_cna"] = truth_cna_file
            calicost_summary["est_cna_file"] = get_calico_cna_file(
                calico_dir, n_cnas, cna_size, ploidy, random
            )

            df_event_f1.append(
                calicost_summary.explode(["event", "F1"]).reset_index(drop=True)
            )

        # Numbat
        numbat_cna_file = get_numbat_cna_file(numbat_dir, simid)

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
            numbat_summary["est_cna_file"] = numbat_cna_file

        df_event_f1.append(
            numbat_summary.explode(["event", "F1"]).reset_index(drop=True)
        )

        # STARCH
        starch_cna_file = get_starch_cna_file(starch_dir, simid)
        starch_gene_cna = read_starch_gene_cna(starch_cna_file)

        F1_dict = compute_gene_F1(true_gene_cna, starch_gene_cna)

        starch_summary = base_summary.copy()
        starch_summary["method"] = "Starch"
        starch_summary["event"] = [list_events]
        starch_summary["F1"] = [[F1_dict[e] for e in list_events]]
        starch_summary["true_cna"] = truth_cna_file
        starch_summary["est_cna_file"] = starch_cna_file

        df_event_f1.append(
            starch_summary.explode(["event", "F1"]).reset_index(drop=True)
        )

    return pd.concat(df_event_f1, ignore_index=True)


def plot_cna_f1s(df_event_f1):
    sim_params = get_sim_params()

    figsize = (
        12,
        4 * (1 + len(sim_params["all_cna_sizes"])),
    )

    fig, axes = plt.subplots(4, 3, figsize=figsize, dpi=300)
    colors = ["#4963C1", "#97b085", "#dad67d"]
    palette = sns.color_palette(colors)

    for i, cnasize in enumerate(sim_params["all_cna_sizes"][::-1]):
        for j, n_cnas in enumerate(sim_params["all_n_cnas"]):
            num_n_cnas = sum(n_cnas)

            isin = df_event_f1.method == "CalicoST"
            isin &= df_event_f1.n_cnas == num_n_cnas
            isin &= df_event_f1.cna_size == __cnasize_mapper[cnasize]

            tmpdf = df_event_f1[isin]

            if len(tmpdf) == 0:
                continue
            """
            sns.scatterplot(
                data=tmpdf,
                x="event",
                y="F1",
                hue="method",
                palette=palette,
                s=10,
                edgecolor="black",
                linewidth=0.5,
                ax=axes[i,j],
            )
            """
            sns.boxplot(
                data=tmpdf,
                x="event",
                y="F1",
                hue="method",
                palette=palette,
                boxprops=dict(alpha=0.7),
                linewidth=1,
                showfliers=False,
                ax=axes[i, j],
            )

            sns.stripplot(
                data=tmpdf,
                x="event",
                y="F1",
                hue="method",
                palette=palette,
                dodge=10,
                edgecolor="black",
                linewidth=0.5,
                ax=axes[i, j],
            )

            axes[i, j].get_legend().remove()
            axes[i, j].set_xlabel(None)
            axes[i, j].set_ylabel(None)
            axes[i, j].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)
            axes[i, j].set_ylim(-0.05, 1.05)

        axes[i, 0].set_ylabel(f"{__cnasize_mapper[cnasize]}" + r" $F_1$")

    h, l = axes[-1, -1].get_legend_handles_labels()
    axes[-1, -1].legend(
        h[:3], l[:3], loc="upper left", bbox_to_anchor=(1, 1), frameon=False
    )

    for j, n_cnas in enumerate(sim_params["all_n_cnas"]):
        axes[0, j].set_title(f"{n_cnas} CNAs")

    fig.tight_layout()
    fig.show()

from __future__ import annotations

from tabarena.models.tabfm.model import TabFMModel
from tabarena.utils.config_utils import ConfigGenerator

# ``.ensemble()`` combines 32 data-view members with an *internal* NNLS/greedy
# weighting fit on TabFM's own cross-validation. TabArena forbids that inner CV
# (AutoGluon already bags + OOF-weights), so instead each member is exposed as
# its own config and AutoGluon's weighted ensemble replaces the NNLS weighting.
#
# ``member_index=i`` runs exactly member ``i`` of the (fully generated) 32-member
# ensemble, so the 32 configs -- sharing ``n_estimators`` / ``random_state`` and
# the ``.ensemble()`` view params -- are bit-identical to the 32 members TabFM
# would build internally. No CV runs inside TabFM (each config is a single view,
# combined by the default mean of one).

_N_MEMBERS = 32
# The view-generation half of ``TabFM*.ensemble()`` (feature crosses + SVD),
# shared by every member so they come from one ensemble context.
_ENSEMBLE_VIEW_PARAMS = {
    "n_estimators": _N_MEMBERS,
    "n_feature_crosses": "sqrt",
    "n_svd_features": "sqrt",
    "random_state": 0,
}


def _ensemble_member_configs(n_members: int = _N_MEMBERS) -> list[dict]:
    return [{**_ENSEMBLE_VIEW_PARAMS, "member_index": i} for i in range(n_members)]


# c1 = plain default TabFM ("TabFM"); c2.. = the 32 ensemble members, which
# AutoGluon's weighted ensemble combines ("TabFM-Ensemble").
gen_tabfm = ConfigGenerator(
    model_cls=TabFMModel,
    manual_configs=[{}, *_ensemble_member_configs()],
    search_space={},
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabfm.generate_all_bag_experiments(
                num_random_configs=0,
            ),
        ),
    )

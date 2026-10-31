from setuptools import find_packages, setup


setup(
    name="dpl-rmf-fast-1d",
    version="2.0.0",
    description="DPL_RMF-fast-1D full-space neural variational RMF/RHF solver",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=("dpl_rhf", "dpl_rhf.*")),
    package_data={"dpl_rhf": ["py.typed"]},
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "matplotlib>=3.7",
        "scipy>=1.10",
        "sympy>=1.12",
    ],
    entry_points={
        "console_scripts": [
            "dpl-rmf-train=dpl_rhf.cli.prl_train:main",
            "dpl-rmf-evaluate=dpl_rhf.cli.evaluate_checkpoint:main",
            "dpl-rmf-batch=dpl_rhf.cli.batch_rmf:main",
            "dpl-rmf-compare=dpl_rhf.cli.compare_core1204:main",
            "dpl-rmf-plot=dpl_rhf.cli.plot_results:main",
            "dpl-rmf-check=dpl_rhf.cli.check_environment:main",
            "dpl-rhf-train=dpl_rhf.cli.rhf_train:main",
            "dpl-rhf-evaluate=dpl_rhf.cli.rhf_evaluate:main",
            "dpl-rhf-compare-core=dpl_rhf.cli.rhf_compare_core:main",
            "dpl-rhf-compare-core1204=dpl_rhf.cli.compare_rhf_core1204:main",
            "dpl-rhf-plot=dpl_rhf.cli.rhf_plot:main",
            "dpl-rhf-batch-pka1=dpl_rhf.cli.batch_pka1_no_refresh:main",
        ]
    },
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# multi_RS1_6x24
rmae_multi_RS1_6x24 = {50: -0.1741660386224395,
                       75: -0.185067864341069,
                       100: -0.18240405862069417,
                       125: -0.18173256404420024,
                       150: -0.1838895482874947,
                       175: -0.17976087163710627,
                       200: -0.18180451419565677
                    }

r2_multi_RS1_6x24 = {50: 0.9095349866672537,
                     75: 0.9048376020457267,
                     100: 0.917778943999547,
                     125: 0.9099127763504142,
                     150: 0.9008595573466824,
                     175: 0.907400227225238,
                     200: 0.9059022664267609
}

# multi_RS2_6x24
rmae_multi_RS2_6x24 = {50: -0.18048840413802722,
                       75: -0.20674719265260985,
                       100: -0.18705392437330254,
                       125: -0.19183027899793037,
                       150: -0.18762863526969725,
                       175: -0.18769637505719516,
                       200: -0.1833982134936633

}
r2_multi_RS2_6x24 = {50: 0.9177272869662664,
                     75: 0.8832739870205065,
                     100: 0.9018579398875545,
                     125: 0.9020273555418361,
                     150: 0.9019051095249621,
                     175: 0.9031099978094972,
                     200: 0.9057931403868373

}

# single_RS1_6x24
rmae_single_RS1_6x24 = {50: -0.2072484796400348,
                        75: -0.217143411013493,
                        100: -0.22143877500139636,
                        125: -0.24142646369852516,
                        150: -0.21637137532003034,
                        175: -0.21438390056798803,
                        200: -0.22453784636843221
                        # 225: -0.22039183960743775,
                        # 250: -0.2326789017301361,
                        # 275: -0.23199699131452184,
                        # 300: -0.23085640109487832,
                        # 330: -0.22950290021256994,
                        # 360: -0.23088127480354706

}
r2_single_RS1_6x24 = {50: 0.8948917488003143,
                      75: 0.8456696304034608,
                      100: 0.8511663178982356,
                      125: 0.8352365888870669,
                      150: 0.8572996183497683,
                      175: 0.8596607914961515,
                      200: 0.8655567409709753
                      # 225: 0.8593631717311683,
                      # 250: 0.852880654389498,
                      # 275: 0.8486328281549914,
                      # 300: 0.854054701691448,
                      # 330: 0.8560002725030367,
                      # 360: 0.8590069972386051

}

# single_RS2_6x24
rmae_single_RS2_6x24 = {50: -0.27407580188409275,
                        75: -0.29595230713883525,
                        100: -0.33794267845925674,
                        125: -0.3327330561098706,
                        150: -0.29259817170756763,
                        175: -0.27263426152284853,
                        200: -0.3182459573006948,
                        # 225: -0.3341866365986222,
                        # 250: -0.3142535094001821,
                        # 275: -0.3079604476204136,
                        # 300: -0.3013527469143015,
                        # 330: -0.3045118395407565,
                        # 360: -0.3199535449042989

}
r2_single_RS2_6x24 = {50: 0.8898789813494246,
                      75: 0.7692050793935622,
                      100: 0.7243378783949427,
                      125: 0.822555444019198,
                      150: 0.8674446980356777,
                      175: 0.8893271525286093,
                      200: 0.7980045195177441
                      # 225: 0.7580652622287879,
                      # 250: 0.8171707903059457,
                      # 275: 0.8417275475354575,
                      # 300: 0.8478985309027527,
                      # 330: 0.8559389554252238,
                      # 360: 0.8185912645132569
}

if __name__ == '__main__':
    # Simulated data for the plot
    steps = r2_multi_RS1_6x24.keys()
    rmaes = {
        "single_RS1_6x24": rmae_single_RS1_6x24.values(),
        "single_RS2_6x24": rmae_single_RS2_6x24.values(),
        "multi_RS1_6x24": rmae_multi_RS1_6x24.values(),
        "multi_RS2_6x24": rmae_multi_RS2_6x24.values(),
    }
    r2s = {
        "single_RS1_6x24": r2_single_RS1_6x24.values(),
        "single_RS2_6x24": r2_single_RS2_6x24.values(),
        "multi_RS1_6x24": r2_multi_RS1_6x24.values(),
        "multi_RS2_6x24": r2_multi_RS2_6x24.values(),
    }

    # Set Seaborn style
    sns.set_style("darkgrid")

    # Plot each method with confidence intervals
    plt.figure(figsize=(5, 4))
    for label, values in rmaes.items():
        # std = std_devs[label]
        plt.plot(steps, values, label=label, alpha=0.6)  # Line plot
        # plt.fill_between(steps, mean - std, mean + std, alpha=0.3)  # Shaded region

    # Add labels, title, and legend
    plt.xlabel("Training set size (schedules)")
    plt.ylabel("Relative MAE")
    plt.title("Relative MAE")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    for label, values in r2s.items():
        # std = std_devs[label]
        plt.plot(steps, values, label=label, alpha=0.6)  # Line plot

    # Add labels, title, and legend
    plt.xlabel("Training set size (schedules)")
    plt.ylabel("R-squared")
    plt.title("R-squared")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


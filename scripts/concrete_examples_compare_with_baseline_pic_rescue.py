import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from io import StringIO

# ================= DATA LOADING =================
# 你提供的新 CSV 数据
csv_data = """RefBaseline,Model,Type,Selection_Method,Selection_Bucket,Selection_Rank,Player,Season,Actual,Base_Pred,Model_Pred,Delta_Pred,Base_Err,Model_Err,Rescue_Amount,Model_Coverage
Stats+Time,Tabular_OnOff,Underrated (Precision),Category,Underrated (Precision),1,Austin Reaves,2024,12976361.999999987,6001125.819730337,10660978.305166828,4659852.485436491,6975236.18026965,2315383.6948331594,4659852.485436491,423
Stats+Time,Tabular_OnOff,Overrated (Precision),Category,Overrated (Precision),2,Kelly Oubre Jr.,2024,7983000.000000004,16652787.741043726,13892707.597292585,-2760080.1437511407,8669787.741043722,5909707.597292582,2760080.1437511407,423
Stats+Time,Tabular_OnOff,Overrated (Overshoot),Category,Overrated (Overshoot),3,Jalen Johnson,2024,4510905.000000001,6984821.494874728,3700388.7265576064,-3284432.7683171215,2473916.494874727,810516.2734423946,1663400.2214323324,423
Stats+Time,RotatE,Underrated (Precision),Category,Underrated (Precision),1,Fred VanVleet,2024,42846615.000000045,7473704.811691208,17448454.193688408,9974749.3819972,35372910.188308835,25398160.806311637,9974749.381997198,423
Stats+Time,RotatE,Underrated (Overshoot),Category,Underrated (Overshoot),2,Dereck Lively II,2024,5014560.000000005,3338661.4440846886,5033070.781702241,1694409.3376175524,1675898.555915316,18510.781702236272,1657387.7742130798,423
Stats+Time,RotatE,Overrated (Precision),Category,Overrated (Precision),3,Kelly Oubre Jr.,2024,7983000.000000004,16652787.741043726,9556269.496305391,-7096518.244738335,8669787.741043722,1573269.4963053875,7096518.244738335,423
Stats+Time,RotatE,Overrated (Overshoot),Category,Overrated (Overshoot),4,Bobby Portis,2024,12578285.999999996,17105356.04193213,11497628.455050794,-5607727.586881334,4527070.041932132,1080657.5449492019,3446412.49698293,423
Stats+Time,Node2Vec,Underrated (Precision),Category,Underrated (Precision),1,Jonathan Isaac,2024,24999999.999999963,4120752.5064734314,8201614.8205473805,4080862.314073949,20879247.493526533,16798385.179452583,4080862.31407395,423
Stats+Time,Node2Vec,Underrated (Overshoot),Category,Underrated (Overshoot),2,Andrew Wiggins,2024,26276786.000000007,22836655.57978264,26930282.155138165,4093626.575355526,3440130.4202173688,653496.1551381573,2786634.2650792114,423
Stats+Time,Node2Vec,Overrated (Precision),Category,Overrated (Precision),3,Kelly Oubre Jr.,2024,7983000.000000004,16652787.741043726,11453809.70259945,-5198978.038444277,8669787.741043722,3470809.7025994454,5198978.038444277,423
Stats+Time,Node2Vec,Overrated (Overshoot),Category,Overrated (Overshoot),4,Bobby Portis,2024,12578285.999999996,17105356.04193213,11701979.805255167,-5403376.236676961,4527070.041932132,876306.1947448291,3650763.847187303,423
Stats+Time,V1,Underrated (Precision),Category,Underrated (Precision),1,Khris Middleton,2024,31666675.999999944,12437976.13653594,24574338.94642553,12136362.80988959,19228699.863464005,7092337.053574413,12136362.809889592,423
Stats+Time,V1,Underrated (Overshoot),Category,Underrated (Overshoot),2,Steven Adams,2024,12600000.000000015,2154461.8874610704,14590572.240455406,12436110.352994336,10445538.112538945,1990572.240455391,8454965.872083554,423
Stats+Time,V1,Overrated (Precision),Category,Overrated (Precision),3,Norman Powell,2024,19241379.0,29358795.565505296,19549467.06235798,-9809328.503147315,10117416.565505296,308088.06235798076,9809328.503147315,423
Stats+Time,V1,Overrated (Overshoot),Category,Overrated (Overshoot),4,Ivica Zubac,2024,11743210.000000015,23484978.492097065,10439196.143173285,-13045782.34892378,11741768.49209705,1304013.85682673,10437754.63527032,423
Stats+Time,V2_Ind,Underrated (Precision),Category,Underrated (Precision),1,Brandon Miller,2024,11424599.999999989,7132892.186177329,11294392.37344722,4161500.1872698916,4291707.81382266,130207.62655276805,4161500.1872698916,423
Stats+Time,V2_Ind,Underrated (Overshoot),Category,Underrated (Overshoot),2,Josh Giddey,2024,8352367.000000005,6403085.7344363425,8543359.589912048,2140273.855475705,1949281.2655636622,190992.58991204295,1758288.6756516192,423
Stats+Time,V2_Ind,Overrated (Precision),Category,Overrated (Precision),3,Ivica Zubac,2024,11743210.000000015,23484978.492097065,17869780.6191098,-5615197.872987267,11741768.49209705,6126570.619109783,5615197.872987267,423
Stats+Time,V2_Ind,Overrated (Overshoot),Category,Overrated (Overshoot),4,Malik Beasley,2024,6000000.0,13755431.349758597,5768465.501905248,-7986965.8478533495,7755431.349758597,231534.49809475243,7523896.851663845,423
Stats+Time,V2_Trans,Underrated (Precision),Category,Underrated (Precision),1,Fred VanVleet,2024,42846615.000000045,7473704.811691208,17378208.912088595,9904504.100397388,35372910.188308835,25468406.08791145,9904504.100397386,423
Stats+Time,V2_Trans,Underrated (Overshoot),Category,Underrated (Overshoot),2,D'Angelo Russell,2024,18692306.999999985,10531889.623491712,20967726.123772107,10435836.500280395,8160417.376508273,2275419.123772122,5884998.252736151,423
Stats+Time,V2_Trans,Overrated (Precision),Category,Overrated (Precision),3,Ivica Zubac,2024,11743210.000000015,23484978.492097065,13450514.28261207,-10034464.209484994,11741768.49209705,1707304.2826120555,10034464.209484994,423
Stats+Time,V2_Trans,Overrated (Overshoot),Category,Overrated (Overshoot),4,Svi Mykhailiuk,2024,3500000.0000000056,7778341.305335629,3147687.7783076144,-4630653.527028015,4278341.305335623,352312.2216923912,3926029.083643232,423
Stats+Time,V2_Full,Underrated (Precision),Category,Underrated (Precision),1,Anthony Edwards,2024,42176400.00000001,27829025.84462184,30614343.271822765,2785317.4272009246,14347374.155378167,11562056.728177242,2785317.4272009246,423
Stats+Time,V2_Full,Underrated (Overshoot),Category,Underrated (Overshoot),2,Cameron Johnson,2024,22500000.00000001,20251897.48347315,22906029.66265789,2654132.1791847423,2248102.516526863,406029.6626578793,1842072.8538689837,423
Stats+Time,V2_Full,Overrated (Precision),Category,Overrated (Precision),3,Jaren Jackson Jr.,2024,25257797.999999996,28136318.054329544,26622255.02111018,-1514063.0332193635,2878520.054329548,1364457.0211101845,1514063.0332193635,423
Stats+Time+Meta,RotatE,Underrated (Precision),Category,Underrated (Precision),1,Fred VanVleet,2024,42846615.000000045,8075345.032977762,17448454.193688408,9373109.160710646,34771269.967022285,25398160.806311637,9373109.160710648,423
Stats+Time+Meta,RotatE,Underrated (Overshoot),Category,Underrated (Overshoot),2,Kyle Filipowski,2024,3000000.000000001,2088361.300343336,3121745.208903312,1033383.9085599761,911638.6996566649,121745.20890331129,789893.4907533536,423
Stats+Time+Meta,RotatE,Overrated (Precision),Category,Overrated (Precision),3,Coby White,2024,12000000.000000007,21455739.147593893,15714914.112883467,-5740825.034710426,9455739.147593886,3714914.11288346,5740825.034710426,423
Stats+Time+Meta,RotatE,Overrated (Overshoot),Category,Overrated (Overshoot),4,Russell Westbrook,2024,3303770.999999999,8328638.987158715,2179935.1174972253,-6148703.8696614895,5024867.987158716,1123835.8825027738,3901032.104655942,423
Stats+Time+Meta,Node2Vec,Underrated (Precision),Category,Underrated (Precision),1,Rudy Gobert,2024,43827586.99999999,21821632.290381357,25701934.24886535,3880301.958483994,22005954.709618635,18125652.75113464,3880301.958483994,423
Stats+Time+Meta,Node2Vec,Underrated (Overshoot),Category,Underrated (Overshoot),2,Trey Murphy III,2024,5159854.999999999,4023407.815354873,5421211.869774993,1397804.0544201196,1136447.184645126,261356.8697749935,875090.3148701326,423
Stats+Time+Meta,Node2Vec,Overrated (Precision),Category,Overrated (Precision),3,Coby White,2024,12000000.000000007,21455739.147593893,16813425.689290185,-4642313.458303709,9455739.147593886,4813425.689290177,4642313.458303709,423
Stats+Time+Meta,Node2Vec,Overrated (Overshoot),Category,Overrated (Overshoot),4,Bobby Portis,2024,12578285.999999996,15756792.63325915,11701979.805255167,-4054812.828003982,3178506.633259153,876306.1947448291,2302200.438514324,423
Stats+Time+Meta,V1,Underrated (Precision),Category,Underrated (Precision),1,Khris Middleton,2024,31666675.999999944,12903569.345914032,24574338.94642553,11670769.600511499,18763106.65408591,7092337.053574413,11670769.600511499,423
Stats+Time+Meta,V1,Underrated (Overshoot),Category,Underrated (Overshoot),2,Steven Adams,2024,12600000.000000015,2436253.2636929736,14590572.240455406,12154318.976762433,10163746.736307042,1990572.240455391,8173174.495851651,423
Stats+Time+Meta,V1,Overrated (Precision),Category,Overrated (Precision),3,Norman Powell,2024,19241379.0,27186826.626819987,19549467.06235798,-7637359.564462006,7945447.626819987,308088.06235798076,7637359.564462006,423
Stats+Time+Meta,V1,Overrated (Overshoot),Category,Overrated (Overshoot),4,Ivica Zubac,2024,11743210.000000015,23258807.746218774,10439196.143173285,-12819611.60304549,11515597.74621876,1304013.85682673,10211583.88939203,423
Stats+Time+Meta,V2_Ind,Underrated (Precision),Category,Underrated (Precision),1,Joel Embiid,2024,51415938.00000004,29237265.87523637,33990135.62629755,4752869.751061179,22178672.124763668,17425802.37370249,4752869.751061179,423
Stats+Time+Meta,V2_Ind,Underrated (Overshoot),Category,Underrated (Overshoot),2,Josh Giddey,2024,8352367.000000005,6982099.719276448,8543359.589912048,1561259.8706355998,1370267.2807235569,190992.58991204295,1179274.690811514,423
Stats+Time+Meta,V2_Ind,Overrated (Precision),Category,Overrated (Precision),3,Ivica Zubac,2024,11743210.000000015,23258807.746218774,17869780.6191098,-5389027.127108976,11515597.74621876,6126570.619109783,5389027.127108976,423
Stats+Time+Meta,V2_Ind,Overrated (Overshoot),Category,Overrated (Overshoot),4,Malik Beasley,2024,6000000.0,13691427.819233408,5768465.501905248,-7922962.317328161,7691427.819233408,231534.49809475243,7459893.321138656,423
Stats+Time+Meta,V2_Trans,Underrated (Precision),Category,Underrated (Precision),1,Fred VanVleet,2024,42846615.000000045,8075345.032977762,17378208.912088595,9302863.879110834,34771269.967022285,25468406.08791145,9302863.879110835,423
Stats+Time+Meta,V2_Trans,Underrated (Overshoot),Category,Underrated (Overshoot),2,D'Angelo Russell,2024,18692306.999999985,11952196.628251523,20967726.123772107,9015529.495520584,6740110.371748462,2275419.123772122,4464691.24797634,423
Stats+Time+Meta,V2_Trans,Overrated (Precision),Category,Overrated (Precision),3,Ivica Zubac,2024,11743210.000000015,23258807.746218774,13450514.28261207,-9808293.463606704,11515597.74621876,1707304.2826120555,9808293.463606704,423
Stats+Time+Meta,V2_Trans,Overrated (Overshoot),Category,Overrated (Overshoot),4,Svi Mykhailiuk,2024,3500000.0000000056,7778341.305335629,3147687.7783076144,-4630653.527028015,4278341.305335623,352312.2216923912,3926029.083643232,423
Stats+Time+Meta,V2_Full,Underrated (Precision),Category,Underrated (Precision),1,Bogdan Bogdanović,2024,17259999.999999978,7760509.935475514,10411096.566662151,2650586.631186637,9499490.064524464,6848903.433337826,2650586.631186638,423
Stats+Time+Meta,V2_Full,Underrated (Overshoot),Category,Underrated (Overshoot),2,Al Horford,2024,9500000.000000015,8381510.843188009,9518830.068132626,1137319.2249446167,1118489.1568120057,18830.06813261099,1099659.0886793947,423
Stats+Time+Meta,V2_Full,Overrated (Precision),Category,Overrated (Precision),3,Malik Monk,2024,17405201.999999996,22950546.82823945,20401539.716348775,-2549007.1118906736,5545344.828239452,2996337.7163487785,2549007.1118906736,423
"""

# 加载数据
df = pd.read_csv(StringIO(csv_data))

# ================= DATA PREPARATION FOR BARS =================
def prepare_data_for_bars(df_subset):
    """
    Converts wide format (Actual, Base_Pred, Model_Pred columns)
    into long format suitable for seaborn barplot hue.
    """
    # 1. Extract Actuals
    df_actual = df_subset[['Player', 'Actual', 'Selection_Rank', 'Model']].copy()
    df_actual = df_actual.rename(columns={'Actual': 'Salary'})
    df_actual['Salary_Type'] = 'Actual Salary'
    # We only need one actual per player, drop duplicates caused by multiple models
    df_actual = df_actual.drop_duplicates(subset=['Player'])

    # 2. Extract Baselines
    df_base = df_subset[['Player', 'Base_Pred', 'Selection_Rank', 'Model']].copy()
    df_base = df_base.rename(columns={'Base_Pred': 'Salary'})
    df_base['Salary_Type'] = 'Baseline Prediction'
    # Similarly, baseline is same for a player given the ref baseline
    df_base = df_base.drop_duplicates(subset=['Player'])

    # 3. Extract Model Predictions (Dynamic Names)
    df_model = df_subset[['Player', 'Model_Pred', 'Model', 'Selection_Rank']].copy()
    df_model = df_model.rename(columns={'Model_Pred': 'Salary'})
    # Create dynamic label like "RotatE Prediction"
    df_model['Salary_Type'] = df_model['Model'] + ' Prediction'

    # 4. Combine and Sort
    df_long = pd.concat([df_actual, df_base, df_model], ignore_index=True)
    # Sort by rank then player to keep them grouped
    df_long = df_long.sort_values(['Selection_Rank', 'Player'])
    return df_long

# ================= COMMON PLOTTING FUNCTION =================
def plot_grouped_bars(df_long, title, filename):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")

    # Define color palette structure
    # Gray for Actual, Red for Baseline, Green for any Model prediction
    palette_map = {
        'Actual Salary': '#95a5a6',
        'Baseline Prediction': '#e74c3c'
    }
    # Dynamically assign green to all model prediction types
    model_types = [t for t in df_long['Salary_Type'].unique() if 'Prediction' in t and 'Baseline' not in t]
    for mt in model_types:
        palette_map[mt] = '#2ecc71'

    # Plot
    ax = sns.barplot(
        data=df_long,
        x='Player',
        y='Salary',
        hue='Salary_Type',
        palette=palette_map,
        edgecolor=".2"
    )

    # Formatting
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'${x/1e6:.0f}M'))
    ax.set_ylabel("Annual Salary (USD, Log Scale)", fontweight='bold')
    ax.set_xlabel("") # Player names are self-explanatory
    ax.set_title(title, fontweight='bold', fontsize=16, y=1.02)
    
    # Rotate x-labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {filename}")
    plt.close()

# ================= MAIN EXECUTION =================

# --- Plot 1: vs. Weak Baseline (Stats+Time) ---
df_weak_wide = df[df['RefBaseline'] == 'Stats+Time'].copy()
# Sort to pick top examples per model for cleaner plot if too many
df_weak_wide = df_weak_wide.sort_values(['Model', 'Selection_Rank']).groupby('Model').head(3)
df_weak_long = prepare_data_for_bars(df_weak_wide)

plot_grouped_bars(
    df_weak_long,
    title="Micro-Level Predictions vs. Weak Baseline (Stats+Time)",
    filename="Figure_3A_Bars_vs_WeakBase.png"
)

# --- Plot 2: vs. Strong Baseline (Stats+Time+Meta) ---
df_strong_wide = df[df['RefBaseline'] == 'Stats+Time+Meta'].copy()
# Sort to pick top examples per model
df_strong_wide = df_strong_wide.sort_values(['Model', 'Selection_Rank']).groupby('Model').head(3)
df_strong_long = prepare_data_for_bars(df_strong_wide)

plot_grouped_bars(
    df_strong_long,
    title="Micro-Level Predictions vs. Strong Baseline (Stats+Time+Meta)",
    filename="Figure_3B_Bars_vs_StrongBase.png"
)

print("\n🎉 Bar charts generated successfully!")
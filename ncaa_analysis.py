from typing import Tuple, List
from os import makedirs
from adjustText import adjust_text
from matplotlib import colormaps as cmp
from sklearn.manifold import TSNE
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True
sns.set_theme(style='whitegrid')  #, palette='viridis')

# %%
FINAL_YEAR = 2024
SEASONS = ['Indoor', 'Outdoor']

def download_flashresults(season: str, final_year: int = FINAL_YEAR, export: bool = False) -> Tuple[pd.DataFrame]:
    '''Download the FlashResults NCAA Championship Results.

    Params:
      -  final_year (`int`): the last year to download (2021 is the earliest, ladf istypically the current year)
      -  season (`str`): the season to download ('Outdoor' or 'Indoor')
      -  export (`bool`): export the data as CSV files to the data directory, default is `False`

    Returns:
      -  data (`Tuple[pd.DataFrame]`): a tuple of the women's and men's dataframes for a season across all selected years.
    '''
    dfs_f = None
    dfs_m = None

    for year in range(2021, final_year + 1):
        if season not in SEASONS:
            raise ValueError('Please choose a valid competition season.')

        root: str = f'https://flashresults.ncaa.com/{season}/{year}/scores_by_event.htm'    
        df_f = pd.read_html(root)[0]
        df_m = pd.read_html(root)[1]

        if df_f is None:
            dfs_f = df_f.assign(year=year, season=season)
            continue
        
        if df_m is None:
            dfs_m = df_m.assign(year=year, season=season)
            continue

        dfs_f = pd.concat([dfs_f, df_f.assign(year=year, season=season)])
        dfs_m = pd.concat([dfs_m, df_m.assign(year=year, season=season)])

    if export:
        makedirs('data', exist_ok=True)
        clean_columns(dfs_f.reset_index(drop=True), sex='f', season=season).to_csv(f'data/DI_women_2021-{year}_{season}.csv', index=False)
        clean_columns(dfs_m.reset_index(drop=True), sex='m', season=season).to_csv(f'data/DI_men_2021-{year}_{season}.csv', index=False)

    return (clean_columns(dfs_f.reset_index(drop=True), sex='f', season=season), 
            clean_columns(dfs_m.reset_index(drop=True), sex='m', season=season))


def clean_columns(df: pd.DataFrame, sex: str, season: str) -> pd.DataFrame:
    '''Rename columns and convert event point columns to floats.

    Params:
      -  df (`pd.DataFrame`): A pandas DataFrame of championship results
      -  sex (`str`): the sex that the championship results table belongs to. Options: 'f' or 'm'
      -  season (`str`): the season to download Options: 'Outdoor' or 'Indoor'

    Returns:
      -  df (`pd.DataFrame`): the input pandas DataFrame reformatted for mathematical operations and future modeling

    '''
    sex_prefix = 'Women' if sex == 'f' else 'Men'
    n_events = 21 if season == 'Outdoor' else 17

    try:
        df = df.drop(columns=['Unnamed: 1'])
        df.rename(columns={'Place': 'place',
                           f'{sex_prefix} [{n_events} of {n_events}]': 'team',
                           'Total': 'total_points'},
                  inplace=True)
    except KeyError:
        df.rename(columns={'Place': 'place',
                           f'{sex_prefix} [{n_events} of {n_events}]': 'team',
                           'Total': 'total_n_events'},
                  inplace=True)

    event_cols = list(df.columns)[3:-2]

    for col in event_cols:
        df.loc[df[col] == '-', col] = float('NaN')
        df[col] = df[col].astype('float')

    return df.reset_index(drop=True)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    '''Add modeling metrics to the dataset. 
    
    General Metrics:
    Discipline Metrics:
    Interdiscipline Metrics:
    Track vs Field Metrics:
    
    Params:
      -  df (`pd.DataFrame`): the dataset to calculate metrics from
      
    Returns:
      -  df (`pd.DataFrame'): the dataset with expanded columns for all the metrics'''

    sprints = ['60', '100', '200', '4x1R']
    # sprints = ['60', '100', '200', '4x1R', '400', '4X4R']
    hurdles = ['60H', '100H', '110H', '400H']
    mid_distance = ['400', '800', '1500', '1Mile', '4X4R']
    # mid_distance = ['800', '1500', '1Mile']
    distance = ['3000', '3000SC', '5000', '10000', 'DMR']
    relays = ['4x1R', '4X4R', 'DMR']
    jumps = ['HJ', 'PV', 'LJ', 'TJ']
    throws = ['SP', 'WT', 'DT', 'HT', 'JT']
    multi = ['Pent', 'Hep', 'Dec']
    track = sprints + hurdles + mid_distance + distance
    field = jumps + throws + multi

    event_cols = list(df.columns)[3:-2]

    def get_columns(discipline: List[str]) -> List[str]:
        return list(filter(lambda x: x in discipline, event_cols))

    def get_meta_columns(metacolumn: List[str]) -> List[str]:
        return list(filter(lambda x: x in metacolumn, list(df.columns)))

    # Overall Metrics
    df['mean_pts_per_event'] = df[event_cols].agg('mean', axis=1)
    df['median_pts_per_event'] = df[event_cols].agg('median', axis=1)
    df['std_pts_per_event'] = df[event_cols].agg('std', axis=1)
    df['n_events_participated'] = df[event_cols].agg('count', axis=1)
    df['n_events_10_pts_or_more'] = df[event_cols].apply(lambda x: x >= 10).agg('sum', axis=1)
    df['n_events_8_pts_or_more'] = df[event_cols].apply(lambda x: x >= 8).agg('sum', axis=1)
    df['n_events_6_pts_or_more'] = df[event_cols].apply(lambda x: x >= 6).agg('sum', axis=1)
    df['n_events_5_pts_or_more'] = df[event_cols].apply(lambda x: x >= 5).agg('sum', axis=1)
    df['n_events_4_pts_or_more'] = df[event_cols].apply(lambda x: x >= 4).agg('sum', axis=1)
    df['n_events_3_pts_or_more'] = df[event_cols].apply(lambda x: x >= 3).agg('sum', axis=1)
    df['n_events_2_pts_or_more'] = df[event_cols].apply(lambda x: x >= 2).agg('sum', axis=1)
    df['n_events_1_pts_or_more'] = df[event_cols].apply(lambda x: x >= 1).agg('sum', axis=1)

    # Intradiscipline metrics
    df['points_sprints'] = df[get_columns(sprints)].agg('sum', axis=1)
    df['mean_pts_sprints'] = df[get_columns(sprints)].agg('mean', axis=1)
    df['median_pts_sprints'] = df[get_columns(sprints)].agg('median', axis=1)
    df['std_pts_sprints'] = df[get_columns(sprints)].agg('std', axis=1)
    df['n_events_sprints'] = df[get_columns(sprints)].agg('count', axis=1)

    df['points_hurdles'] = df[get_columns(hurdles)].agg('sum', axis=1)
    df['mean_pts_hurdles'] = df[get_columns(hurdles)].agg('mean', axis=1)
    df['median_pts_hurdles'] = df[get_columns(hurdles)].agg('median', axis=1)
    df['std_pts_hurdles'] = df[get_columns(hurdles)].agg('std', axis=1)
    df['n_events_hurdles'] = df[get_columns(hurdles)].agg('count', axis=1)

    df['points_mid_distance'] = df[get_columns(mid_distance)].agg('sum', axis=1)
    df['mean_pts_mid_distance'] = df[get_columns(mid_distance)].agg('mean', axis=1)
    df['median_pts_mid_distance'] = df[get_columns(mid_distance)].agg('median', axis=1)
    df['std_pts_mid_distance'] = df[get_columns(mid_distance)].agg('std', axis=1)
    df['n_events_mid_distance'] = df[get_columns(mid_distance)].agg('count', axis=1)

    df['points_distance'] = df[get_columns(distance)].agg('sum', axis=1)
    df['mean_pts_distance'] = df[get_columns(distance)].agg('mean', axis=1)
    df['median_pts_distance'] = df[get_columns(distance)].agg('median', axis=1)
    df['std_pts_distance'] = df[get_columns(distance)].agg('std', axis=1)
    df['n_events_distance'] = df[get_columns(distance)].agg('count', axis=1)

    df['points_relays'] = df[get_columns(relays)].agg('sum', axis=1)
    df['mean_pts_relays'] = df[get_columns(relays)].agg('mean', axis=1)
    df['median_pts_relays'] = df[get_columns(relays)].agg('median', axis=1)
    df['n_events_relays'] = df[get_columns(relays)].agg('count', axis=1)

    df['points_jumps'] = df[get_columns(jumps)].agg('sum', axis=1)
    df['mean_pts_jumps'] = df[get_columns(jumps)].agg('mean', axis=1)
    df['median_pts_jumps'] = df[get_columns(jumps)].agg('median', axis=1)
    df['std_pts_jumps'] = df[get_columns(jumps)].agg('std', axis=1)
    df['n_events_jumps'] = df[get_columns(jumps)].agg('count', axis=1)

    df['points_throws'] = df[get_columns(throws)].agg('sum', axis=1)
    df['mean_pts_throws'] = df[get_columns(throws)].agg('mean', axis=1)
    df['median_pts_throws'] = df[get_columns(throws)].agg('median', axis=1)
    df['std_pts_throws'] = df[get_columns(throws)].agg('std', axis=1)
    df['n_events_throws'] = df[get_columns(throws)].agg('count', axis=1)

    df['points_multi'] = df[get_columns(multi)].agg('sum', axis=1)
    df['mean_pts_multi'] = df[get_columns(multi)].agg('mean', axis=1)
    df['median_pts_multi'] = df[get_columns(multi)].agg('median', axis=1)
    df['n_events_multi'] = df[get_columns(multi)].agg('count', axis=1)

    df['points_track'] = df[get_columns(track)].agg('sum', axis=1)
    df['mean_pts_track'] = df[get_columns(track)].agg('mean', axis=1)
    df['median_pts_track'] = df[get_columns(track)].agg('median', axis=1)
    df['std_pts_track'] = df[get_columns(track)].agg('std', axis=1)
    df['n_events_track'] = df[get_columns(track)].agg('count', axis=1)

    df['points_field'] = df[get_columns(field)].agg('sum', axis=1)
    df['mean_pts_field'] = df[get_columns(field)].agg('mean', axis=1)
    df['median_pts_field'] = df[get_columns(field)].agg('median', axis=1)
    df['std_pts_field'] = df[get_columns(field)].agg('std', axis=1)
    df['n_events_field'] = df[get_columns(field)].agg('count', axis=1)

    # Interdiscipline Metrics
    discipline_points = ['points_sprints', 'points_hurdles', 'points_mid_distance',
                        'points_distance', 'points_relays', 'points_jumps',
                        'points_throws', 'points_multi']
    discipline_n_events = ['n_events_sprints', 'n_events_hurdles', 'n_events_mid_distance',
                        'n_events_distance', 'n_events_relays', 'n_events_jumps',
                        'n_events_throws', 'n_events_multi']

    df['mean_pts_disciplines'] = df[get_meta_columns(discipline_points)].agg('mean', axis=1)
    df['median_pts_disciplines'] = df[get_meta_columns(discipline_points)].agg('median', axis=1)
    df['std_pts_disciplines'] = df[get_meta_columns(discipline_points)].agg('std', axis=1)

    df['mean_n_disciplines'] = df[get_meta_columns(discipline_n_events)].agg('mean', axis=1)
    df['median_n_disciplines'] = df[get_meta_columns(discipline_n_events)].agg('median', axis=1)
    df['std_n_disciplines'] = df[get_meta_columns(discipline_n_events)].agg('std', axis=1)

    df['n_disciplines_10_pts_or_more'] = df[get_meta_columns(discipline_points)].apply(lambda x: x >= 10).agg('sum', axis=1)

    # Track vs Field Metrics
    df['track_vs_field_differential'] = df['points_track'] - df['points_field']  # Negative = more field points
    df['track_vs_field_mean_differential'] = df['mean_pts_track'] - df['mean_pts_field']  # Negative = more field points
    df['track_vs_field_n_differential'] = df['n_events_track'] - df['n_events_field']  # Negative = compete more field events
    df['track_vs_field_track_proportion'] = df['points_track'] / df['total_points']
    df['track_vs_field_field_proportion'] = 1 - df['track_vs_field_track_proportion']

    # Place modification for plotting
    df['log_1_place'] = np.log(1 / df['place'])

    return df.fillna(0)


def perform_TSNE(df: pd.DataFrame) -> Tuple[TSNE, pd.DataFrame]:
    df[['tsne_x', 'tsne_y']] = TSNE(n_components=2, 
                                    learning_rate='auto',
                                    perplexity=45,
                                    init='pca', 
                                    random_state=69420).fit_transform(df.select_dtypes(['float', 'int']))
    return df


def visualize_TSNE_place_double(df: pd.DataFrame, suptitle: str | None = None) -> None:
    plt.figure(figsize=(12,8))
    plt.suptitle(f'$t$-SNE: {suptitle}')

    plot1_cmap = cmp['viridis'].reversed()
    plt.subplot(1,2,1)
    plt.scatter(x=df['tsne_x'], 
                y=df['tsne_y'], 
                c=df['place'], 
                cmap=plot1_cmap)
    plt.colorbar(orientation='horizontal', 
                pad=0.01,
                label='Team Place')
    plt.title('Colored according to Place')
    plt.tick_params(left=None, labelleft=False,
                    bottom=None, labelbottom=False)

    plt.subplot(1,2,2)
    plt.scatter(x=df['tsne_x'], 
                y=df['tsne_y'], 
                c=df['log_1_place'], 
                cmap='viridis')
    plt.colorbar(orientation='horizontal', 
                pad=0.01,
                label=r'$log(\frac{1}{Place})$')
    plt.title('Colored to accentuate the top 3 places')
    plt.tick_params(left=None, labelleft=False,
                    bottom=None, labelbottom=False)

    plt.subplots_adjust(wspace=0.01)

    plt.show()
    return


def visualize_TSNE_points(df: pd.DataFrame, title: str | None = None) -> None:
    plt.figure(figsize=(7,8))
    plt.scatter(x=df['tsne_x'], 
                y=df['tsne_y'], 
                c=df['total_points'], 
                cmap='viridis')
    plt.colorbar(orientation='horizontal', 
                pad=0.01,
                label='Team Points')
    plt.title(f'$t$-SNE: {title}\nHighlighted by Total Points')
    plt.tick_params(left=None, labelleft=False,
                    bottom=None, labelbottom=False)
    plt.show()
    return


def visualize_TSNE_point_proportions(df: pd.DataFrame, title: str | None = None, label: bool = False) -> None:
    plt.figure(figsize=(7,8))
    plt.scatter(x=df['tsne_x'], 
                y=df['tsne_y'], 
                c=df['track_vs_field_track_proportion'], 
                cmap='viridis')
    plt.colorbar(orientation='horizontal', 
                pad=0.01,
                label='Proportion of Team Points from Track Events')
    plt.suptitle(f'$t$-SNE: {title}\nHighlighted by Specialization in Track vs Field', 
                y=0.98)
    plt.title('0 (Blue) = All points from Field Events\n1 (Yellow) = All points from Track Events',
              fontsize='small')
    plt.tick_params(left=None, labelleft=False,
                    bottom=None, labelbottom=False)
    
    if label:
        text_labels = []
        for idx, row in df.iterrows():
            if row['total_points'] >= 40:
                text_labels.append(plt.text(row['tsne_x'], 
                                            row['tsne_y'], 
                                            f'''{idx}: {row['place']}''',
                                            c='black'))
        adjust_text(text_labels, 
                    only_move={'points': 'y', 
                               'texts': 'xy'},
                    arrowprops=dict(arrowstyle="->", color='r', lw=1.5),
                    force_text=(4,5))

    plt.show()
    return


def visualize_top_10_over_time(df: pd.DataFrame, title: str | None = None) -> None:
    ts_df = df.loc[df['place'] <= 10].reset_index(drop=True)
    color_vals = cmp['viridis'].reversed()(np.linspace(0, 1, num=10))

    plt.figure(figsize=(6*1.618, 6))
    ts_plot = sns.lineplot(x='year', 
                           y='total_points', 
                           hue='place', 
                           style='place',
                           markers=True,
                           data=ts_df, 
                           palette=list(color_vals))
    ts_plot.set_xticks(range(ts_df['year'].min(), 2025, 1))
    ts_plot.set_title(title, 
                    fontsize=20, 
                    y=1.05,
                    weight='bold')
    sns.move_legend(obj=ts_plot, loc='upper left', bbox_to_anchor=(1, 1))

    ts_labels=[]
    for idx, row in ts_df.iterrows():
        ts_labels.append(plt.text(row['year'], 
                                  row['total_points'], 
                                  row['team'],
                                  c='black'))
    adjust_text(ts_labels, 
                only_move={'points': 'xy', 
                           'texts': 'xy'},
                arrowprops=dict(arrowstyle="->", color='r', lw=2),
                force_text=(1,1))

    plt.show()
    return


def correlate_and_plot_events(df: pd.DataFrame, method: str = 'spearman') -> pd.DataFrame:
    n_cols = ['total_points', 'place'] + \
         [col for col in df.columns if 'n_e' in col] + \
         ['n_disciplines_10_pts_or_more', 'mean_n_disciplines']
    
    n_events_cols = [
    'total_points',
    'log_1_place',
    'n_events_participated',
    'n_events_8_pts_or_more',
    'n_events_6_pts_or_more',
    'n_events_5_pts_or_more',
    'mean_n_disciplines',
    'n_disciplines_10_pts_or_more',
    ]   

    sns.pairplot(df[n_events_cols], 
                hue='log_1_place', 
                palette='viridis',
                y_vars='total_points',
                x_vars=['n_events_8_pts_or_more',
                        'n_events_6_pts_or_more',
                        'n_events_5_pts_or_more'])
    plt.show()

    sns.pairplot(df[n_events_cols], 
                hue='log_1_place', 
                palette='viridis', 
                y_vars='total_points',
                x_vars=['n_events_participated',
                        'mean_n_disciplines',
                        'n_disciplines_10_pts_or_more'])
    plt.show()

    sns.pairplot(df[['total_points', 'log_1_place', 
                        'track_vs_field_n_differential', 'track_vs_field_track_proportion', 
                        'track_vs_field_differential']],
                hue='log_1_place',
                y_vars='total_points',
                x_vars=['track_vs_field_n_differential'],
                palette='viridis')
    plt.show()
    
    return df[n_cols].corr(method=method)[['total_points', 'place']].iloc[2:]\
                     .style.set_caption('Spearman Correlation Coefficients') \
                     .set_table_styles([
                                 {
                                     'selector': 'caption',
                                     'props': [
                                         ('font-family', 'Franklin Gothic Book'),
                                         ('font-weight', 'bold'),
                                         ('font-size', '20px'),
                                         ('margin', '0 0 10px 0')
                                         ]
                                 }
                     ]) \
                     .format(precision=2)


def correlate_and_plot_points(df: pd.DataFrame, method: str = 'spearman') -> pd.DataFrame:
    std_cols = ['total_points', 'place'] + [col for col in df.columns if 'std' in col] + ['mean_pts_disciplines']
    std_corrs = df[std_cols].corr(method='spearman')[['total_points', 'place']].iloc[2:]

    std_pts_cols = [
    'total_points',
    'log_1_place',
    'std_pts_per_event',
    'std_pts_disciplines',
    'std_n_disciplines',
    'std_pts_track',
    'std_pts_field',  
    'mean_pts_disciplines', 
    ]


    sns.pairplot(df[std_pts_cols],
                y_vars='total_points', 
                x_vars=['std_pts_per_event',
                        'std_pts_disciplines',
                        'std_n_disciplines'],
                hue='log_1_place', 
                palette='viridis')
    plt.show()

    sns.pairplot(df[std_pts_cols],
                y_vars='total_points', 
                x_vars=['std_pts_track',
                        'std_pts_field',  
                        'mean_pts_disciplines',],
                hue='log_1_place', 
                palette='viridis')
    plt.show()

    sns.pairplot(df[['total_points', 'log_1_place', 
                        'track_vs_field_n_differential', 'track_vs_field_track_proportion', 
                        'track_vs_field_differential', 'n_events_participated', 'mean_pts_per_event']],
                hue='log_1_place',
                y_vars='total_points',
                x_vars=['track_vs_field_differential', 'track_vs_field_track_proportion', 'mean_pts_per_event'],
                palette='viridis')
    plt.show()


    return std_corrs.style.set_caption('Spearkman Correlation Coefficients') \
                          .set_table_styles([
                                      {
                                          'selector': 'caption',
                                          'props': [
                                              ('font-family', 'Franklin Gothic Book'),
                                              ('font-weight', 'bold'),
                                              ('font-size', '20px'),
                                              ('margin', '0 0 10px 0')
                                              ]
                                      }
                          ]) \
                          .format(precision=2)


def perform_basic_analysis(df: pd.DataFrame, title: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = perform_TSNE(df)

    visualize_TSNE_place_double(df=df, suptitle=title)
    visualize_TSNE_points(df=df, title=title)
    visualize_TSNE_point_proportions(df=df, title=title)
    visualize_top_10_over_time(df=df, title=title)

    corr_table_events = correlate_and_plot_events(df=df)
    corr_table_points = correlate_and_plot_points(df=df)

    return corr_table_events, corr_table_points
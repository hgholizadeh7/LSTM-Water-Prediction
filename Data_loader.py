from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class AlabamaGWL(BaseDataset):
  

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 well: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(AlabamaGWL, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       well=well,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_well_data(self, well: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df, area = load_Alabama_gwl_forcings(self.cfg.data_dir, well, forcing)

            # rename columns
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # add gwd
        df['GWDobs(m)'] = load_Alabama_gwl_gwd(self.cfg.data_dir, well, area)

        # replace invalid gwd values by NaNs
        gwdobs_cols = [col for col in df.columns if "gwdobs" in col.lower()]
        for col in gwdobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_Alabama_gwl_attributes(self.cfg.data_dir, wells=self.wells)


def load_Alabama_gwl_attributes(data_dir: Path, wells: List[str] = []) -> pd.DataFrame:
  
    attributes_path = Path(data_dir) / 'Alabama_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('Alabama_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'well_id': str})
        df_temp = df_temp.set_index('well_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if wells:
        if any(b not in df.index for b in wells):
            raise ValueError('Some wells are missing static attributes.')
        df = df.loc[wells]

    return df


def load_Alabama_gwl_forcings(data_dir: Path, well: str, forcings: str) -> Tuple[pd.DataFrame, int]:

    forcing_path = data_dir / 'well_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'**/{well}_*_forcing_leap.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for well {well} at {file_path}')


    with open(file_path, 'r') as fp:
        # load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # load the dataframe from the rest of the well
        df = pd.read_csv(fp, sep='\s+')
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index("date")

    return df, area


def load_Alabama_gwl_gwd(data_dir: Path, well: str, area: int) -> pd.Series:


    gwd_path = data_dir / 'Alabama_GWD'
    file_path = list(gwd_path.glob(f'**/{well}_GWD.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for well {well} at {file_path}')

    col_names = ['well', 'Year', 'Mnth', 'Day', 'gwdobs']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

   

    return df.gwdobs

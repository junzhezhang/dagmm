import numpy as np 
import pandas as pd
from sklearn.preprocessing import minmax_scale


def load_data_v2(file_path, cols=None, scale = (0,1)):
    """
    scalce can chose from (0,1) and (-1,1) for numerical cols. 
    dagmm is using (0,1) normalization
    in v2, data contains labels. 
    
    """
    COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels"]

    data = pd.read_csv(file_path, names=COL_NAMES, index_col=False)
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    NOM_IND = [1, 2, 3]
    BIN_IND = [6, 11, 13, 14, 20, 21]
    # Need to find the numerical columns for normalization
    NUM_IND = list(set(range(40)).difference(NOM_IND).difference(BIN_IND))

    # Scale all numerical data to [0-1]
    data.iloc[:, NUM_IND] = minmax_scale(data.iloc[:, NUM_IND],feature_range=scale)
    labels = data['labels']
    data.loc[data["labels"] != "normal", 'labels'] = 0
    data.loc[data["labels"] == "normal", 'labels'] = 1
    # Binary labeling
#     del data['labels']
    data = pd.get_dummies(data)
    sorted_cols = list(set(data.columns) - set(['labels']))+['labels']
    if cols is None:
        cols = sorted_cols
        data = data[cols]
#         cols = data.columns
    else:
        map_data = pd.DataFrame(columns=cols)
        map_data = map_data.append(data)
        data = map_data.fillna(0)
        data = data[cols]

    return [data, cols]

train, cols = load_data_v2('nsl_kdd_train.csv')

test, clos_2 = load_data_v2('nsl_kdd_test.csv',cols)



np.savez_compressed("nsl_kdd_train",nsl_kdd_train=train.as_matrix())

np.savez_compressed("nsl_kdd_test",nsl_kdd_test=test.as_matrix())
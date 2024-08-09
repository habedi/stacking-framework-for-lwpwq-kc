import pandas as pd

from l0_settings import Settings


class Helper:
    @staticmethod
    def fix_encoding(input_str):
        replacements = {
            'Ä±': '1', 'Â´': "'", '\x97': '', 'Ë\x86': 't',
            'Å\x9f': 'DollarsSymbol', '\x9b': '', '\x96': '-',
            'â\x80\x93': '-', '\x80': 'EurosSymbol', 'ä': 'a',
            '¡': 'i', '¿': '?'
        }

        for original_char, replacement_char in replacements.items():
            input_str = input_str.replace(original_char, replacement_char)

        return input_str.encode('latin-1', 'replace').decode('latin-1')

    @staticmethod
    def replace_move(activity_name):
        return 'Replace' if 'move' in activity_name.lower() else activity_name


global_settings = Settings()

train_logs = pd.read_csv(global_settings.get_train_logs_path())
test_logs = pd.read_csv(global_settings.get_test_logs_path())

print('Fixing encoding for train logs')
train_logs['down_event'] = train_logs['down_event'].apply(Helper.fix_encoding)
train_logs['up_event'] = train_logs['up_event'].apply(Helper.fix_encoding)
train_logs['activity'] = train_logs.activity.apply(lambda x: Helper.replace_move(x))

print('Fixing encoding for test logs')
test_logs['down_event'] = test_logs['down_event'].apply(Helper.fix_encoding)
test_logs['up_event'] = test_logs['up_event'].apply(Helper.fix_encoding)
test_logs['activity'] = test_logs.activity.apply(lambda x: Helper.replace_move(x))

print('Saving fixed logs...')
# train_logs.to_csv(global_settings.get_train_logs_path('train_logs_fixed.csv.gz'), index=False)
# test_logs.to_csv(global_settings.get_test_logs_path('test_logs_fixed.csv.gz'), index=False)

train_logs.to_csv(global_settings.get_train_logs_path('train_logs.csv.gz'), index=False)
test_logs.to_csv(global_settings.get_test_logs_path('test_logs.csv.gz'), index=False)

print('Done!')

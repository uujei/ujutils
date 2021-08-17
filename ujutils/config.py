
# Extensions
EXTS_IMAGE = ['jpg', 'png', 'gif', 'bmp', 'tiff']
EXTS_SIGNAL = ['wav', 'mp3', 'tmds']
EXTS_TEXT = ['csv', 'txt', 'xls', 'xlsx']

# Oder of ML related wrods - for sorting
ML_WORD_ORDER = [
    ['train'],
    ['val', 'valid', 'validation'],
    ['dev', 'develop', 'deveopment'],
    ['test'],
    ['eval', 'evaluate', 'evaluation'],
    ['ok', 'neg', 'nega', 'negative'],
    ['ng', 'pos', 'posi', 'positive'],
]

# Styles for inquirer
POINTER = '> '
ENABLED_SYMBOL = '● '
DISABLED_SYMBOL = '○ '
_STYLE = {
    'pointer': POINTER
}
CHECKBOX_STYLE = {
    'pointer': POINTER,
    'enabled_symbol': ENABLED_SYMBOL,
    'disabled_symbol': DISABLED_SYMBOL,
}

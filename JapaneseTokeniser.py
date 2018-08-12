# -*- coding: utf-8 -*-
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter


class JapaneseTokeniser:
    """
    This class tokenises Japanese sentences, removing designed part of speech.

    Referenced URLs to implement the class:
    このクラスを実装するに当たり参照したURLはこちら：
    # Python初心者が1時間以内にjanomeで形態素解析できた方法
    https://qiita.com/d-cabj/items/d934eb87e3012a02e23a
    # はしくれエンジニアもどきのメモ
    http://cartman0.hatenablog.com/entry/2017/11/21/Python%E3%81%A7Janome%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90
    Janome v0.3 documentation (日本語)
    http://mocobeta.github.io/janome/
    """

    def __init__(self):
        self.t = Tokenizer()
        self.token_filters = [POSKeepFilter(['名詞', '動詞', '形容詞', '形容動詞', '接続詞'])]
        self.char_reg_filter = [("[,\.\(\)\{\}\[\]]", " ")]

    def analyse_japanese(self, list_speech):
        char_filters = [UnicodeNormalizeCharFilter()]
        for reg in self.char_reg_filter:
            char_filters.append(RegexReplaceCharFilter(*reg))
        a2 = Analyzer(char_filters, self.t, self.token_filters)
        list_token = []
        for item in list_speech:
            list_token.append([str(token.base_form) for token in a2.analyze(item)])
        list_str = []
        for j in list_token:
            str2 = ""
            for i in j:
                str2 += i + ' '
            list_str.append(str2)
        return list_str



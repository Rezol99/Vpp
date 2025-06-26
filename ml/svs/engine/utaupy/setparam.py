#! /usr/bin/env python3
# coding: utf-8
# Copyright (c) 2020 oatsu
"""
setParamプラグイン用モジュール
## リンク
- setParam: http://nwp8861.web.fc2.com/soft/setParam
- setParamプラグインの入出力ファイルの仕様: http://nwp8861.blog92.fc2.com/blog-entry-365.html
"""

from . import otoini


def run(your_function):
    """
    setParamプラグイン用ファイルの入出力をする。
    """
    with open("inparam.txt", "r") as f_ip:
        lines = [line.rstrip("\n") for line in f_ip.readlines]
        # プラグイン入出力用ファイル 'oto-autoEstimation.ini' のパス
        path_oto_autoestimation_ini = lines[1]
    # utaupy.otoini.OtoIni クラスインスタンスにする
    oto_autoestimation_ini = otoini.load(path_oto_autoestimation_ini)
    # やりたい処理を実行
    your_function(oto_autoestimation_ini)
    # 上書き保存
    oto_autoestimation_ini.write(path_oto_autoestimation_ini)

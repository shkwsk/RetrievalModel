# -*- coding: utf-8 -*-
#!/usr/local/bin/python2.7
"""
vector,token,typeファイルから、ベクトル空間モデルで文書類似度を計算するクラス
"""
import sys, codecs, argparse, json, time, math
from collections import defaultdict

###############################################################################################
class VSM(object):
    def __init__(self):
        self.score = defaultdict(dict)

    @staticmethod
    def __idf(num_coll, veclen, docs):
        """ idf(文書頻度の逆数)ベクトルを返す
        
        num_coll -- コレクション内の文書数
        veclen -- クエリベクトル長
        docs -- クエリに対する文書ベクトルの集合
        """
        df = [0] * veclen
        for vd in docs.values():
            for i, vdi in enumerate(vd):
                if vdi: df[i] += 1
        idf = [0] * veclen
        for i, dfi in enumerate(df):
            idf[i] = math.log( float(num_coll) / dfi, 2 )
        return idf

    def InnerProduct(self, vectors):
        """ クエリと文書のベクトルを内積する

        vectors -- クエリと文書のベクトル
        """
        for q_id, q_data in sorted(vectors.items()):
            for d_id, vd in q_data['docs'].items():
                inprod = 0.0
                for i, (vqi, vdi) in enumerate(zip(q_data['q_vec'], vd)):
                    inprod += vqi * vdi
                self.score[q_id].update({ d_id:inprod })

    def InnerProduct_log(self, vectors):
        """ クエリと文書のベクトルを内積する """
        for q_id, q_data in sorted(vectors.items()):
            for d_id, vd in q_data['docs'].items():
                inprod = 0.0
                for i, (vqi, vdi) in enumerate(zip(q_data['q_vec'], vd)):
                    inprod += vqi * math.log(1 + vdi, 2)
                self.score[q_id].update({ d_id:inprod })

    def InnerProduct_IDF(self, vectors, num_coll):
        """ クエリと文書のベクトルを内積する

        vectors -- クエリと文書のベクトル
        num_coll -- 文書総数
        """
        for q_id, q_data in sorted(vectors.items()):
            idf = self.__idf( num_coll, len(q_data['q_vec']), q_data['docs'] )
            for d_id, vd in q_data['docs'].items():
                inprod = 0.0
                for i, (vqi, vdi) in enumerate(zip(q_data['q_vec'], vd)):
                    inprod += vqi * vdi * idf[i]
                self.score[q_id].update({ d_id:inprod })

    def InnerProduct_log_IDF(self, vectors, num_coll):
        """ クエリと文書のベクトルを内積する """
        for q_id, q_data in sorted(vectors.items()):
            idf = self.__idf( num_coll, len(q_data['q_vec']), q_data['docs'] )
            for d_id, vd in q_data['docs'].items():
                inprod = 0.0
                for i, (vqi, vdi) in enumerate(zip(q_data['q_vec'], vd)):
                    inprod += vqi * math.log(1 + vdi, 2) * idf[i]
                self.score[q_id].update({ d_id:inprod })

    def Normalization_DocLen(self, tokens):
        """ 文書長でスコアを正規化する """
        for q_id, docs in self.score.items():
            for d_id, score in docs.items():
                self.score[q_id][d_id] = float(score) / tokens[d_id]

    def Normalization_Pivot(self, types, sigma):
        """ スコアをピボット正規化する """
        ave_types = float(sum([ f for f in types.values() ])) / len(types)
        for q_id, docs in self.score.items():
            for d_id, score in docs.items():
                norm = (1 - sigma) * ave_types + sigma * types[d_id] # ピボット正規化項
                self.score[q_id][d_id] = float(score) / norm

# -*- coding: utf-8 -*-
#!/usr/local/bin/python2.7
"""
vector,token,typeファイルから、ベクトル空間モデルで文書類似度を計算するクラス
"""
import sys, codecs, argparse, json, time, math, itertools
from collections import defaultdict

###############################################################################################
class COOC(object):
    def __init__(self):
        pass

    @staticmethod
    def __removeZeroVector(q_vec, docs):
        """ コレクションに現れないクエリベクトルを除去する """
        check_remove = [0] * len(q_vec)
        for vd in docs.values():
            for i, vdi in enumerate(vd):
                check_remove[i] += vdi
        remove_idx = []
        for i, chk in enumerate(check_remove):
            if chk == 0: remove_idx.append(i - len(remove_idx))
        for i in remove_idx:
            q_vec.pop(i)
            for vd in docs.values():
                vd.pop(i)
        return q_vec, docs

    def generateCooccurrenceVector(self, vectors):
        """ 共起ベクトルを生成する """
        c_vectors = defaultdict(dict)
        for q_id, q_data in vectors.items():
            c_vectors[q_id] = defaultdict(dict)
            cq = [1] * len(list(itertools.combinations(q_data['q_vec'], 2)))
            cds = {}
            for d_id, vd in q_data['docs'].items():
                cds.update({ d_id:[ int(x > 0 and y > 0) \
                                        for x, y in list(itertools.combinations(vd, 2)) ] })
            cq, cds = self.__removeZeroVector( cq, cds )
            c_vectors[q_id].update({ 'q_vec': cq  })
            c_vectors[q_id].update({ 'docs' : cds })
        return c_vectors
        
    def generateCooccurrenceNorm(self, norms):
        """ 各文書の共起単語のトークン数、タイプ数を計算する """
        c_norms = {}
        for d_id, norm in norms.items():
            c_norms.update({ d_id : (norm*(norm -1))/2 })
        return c_norms

    @staticmethod
    def MixingScore(score, c_score, cooc):
        """ クエリ尤度スコアと文書信頼度を線形補間する """
        mix_score = defaultdict(dict)
        q_ids = [ q_id for q_id in score.keys() ]
        d_ids = {}
        for q_id in q_ids:
            d_ids.update({ q_id:set([ d_id for d_id in score[q_id].keys() ] + [ d_id for d_id in c_score[q_id].keys() ]) })
        for q_id in q_ids:
            for d_id in d_ids[q_id]:
                mix_score[q_id].update({ d_id : (1 - cooc) * score[q_id].get(d_id, 0) + cooc * c_score[q_id].get(d_id, 0) })
        return mix_score

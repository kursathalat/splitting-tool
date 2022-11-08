#Sentence Split on ML model
#Abbreviations are Turkish-only

import numpy as np
from nltk import tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re


abbreviations = {'a.', 'Adr.', 'Alb.', 'Alm.', 'Ank.', 'Apt.', 'Ar.', 'As.', 'As. İz.', 'Asb.', 'Ass.', 'Atgm.', 'atm.', 'Av.', 'B.', 'bak.', 'bas.', 'Bçvş.', 'bkz.', 'Bl.', 'Bl. K.', 'Bn.', 'Bnb.', 'böl.', 'Bşk.', 'Bştbp.', 'Bul.', 'c.', 'Cad.', 'cal.', 'Cum. Başk.', 'Çev.', 'Çvş.', 'dak.', 'Der.', 'dilb.', 'Dipl.', 'Doç.', 'doğ.', 'Dr.', 'Dz. Ataş.', 'Dz. K.', 'Dz. Kuv.', 'Dz. Kuv. K.', 'Ecz.', 'ed.', 'Ed. F.', 'Ens.', 'Erm.', 'F.', 'Fak.', 'Far.', 'fiz.', 'folk.', 'foto.', 'Fr.', 'Gen.', 'Gmr.', 'Gn.Kur.', 'gön.', 'H.', 'hlk.', 'Hrk. Başk.', 'Hrp. Ak.', 'Hrp. TD. Başk.', 'Hs. Uz.', 'Hst.', 'Hv.', 'Hv. Kuv.', 'Hz.', 'İbr.', 'İmp.', 'İng.', 'İrt. Sb.', 'İst.', 'İşl.', 'İt.', 'İzm.', 'J.', 'Jap.', 'K.', 'K. Kuv.', 'Kd.', 'Kol. Şti.', 'Kom.', 'Kom. Şti.', 'Kor.', 'Kora.', 'Korg.', 'krş.', 'Kur.', 'Kur. Bşk.', 'Kuv.', 'Lat.', 'lt.', 'Ltd.', 'Lv.', 'mad.', 'Mah.', 'mat.', 'mey.', 'Mim.', 'Mrşl.', 'Müd.', 'Müh.', 'No.', 'Nö. Amr.', 'Nöb.', 'Odt.', 'Onb.', 'Opr.', 'Or.', 'Or. Gn. Kh.', 'Ora.', 'Ord.', 'Org.', 'Ort.', 'Osm.', 'oto.', 'Öğr.', 'ölm.', 'örn.', 'Plt.', 'Por.', 'Prof.', 'Rus.', 's.', 'sa.', 'sat.', 'Sb.', 'Sn.', 'Sok.', 'Sr.', 'süt.', 'Şb.', 'Şrt.', 'Şti.', 'Tb.', 'Tbp.', 'Tel.', 'Telg.', 'Tğm.', 'Tic.', 'tiy.', 'Top.', 'Tug.', 'Tuğa.', 'Tuğg.', 'Tüm.', 'Tüma.', 'Tümg.', 'ui.', 'Uz. J.', 'Uzm.', 'Üçvş.', 'Üni.', 'Ütğm.', 'vb.', 'Vet.', 'vö.', 'vs.', 'Y. Müh.', 'Yard.', 'Yard. Doç.', 'Yay.', 'yay. haz.', 'Yd.', 'Yd. Sb.', 'Yrb.', 'Yun.', 'yy.', 'Yzb.'}

f = open("text.txt")
corpus = f.read()

def sent_features(data):
    features = np.empty([0, 5])

    sentences = tokenize.sent_tokenize(data)
    sentences = [tokenize.word_tokenize(sent) for sent in sentences]

    tokens = []
    length = 0
    boundary = set()

    for sentence in sentences:
        for t in sentence:
            if "." in t and len(t) > 1:
                index_match = sentence.index(t)
                sentence = sentence[:index_match] + list(t.partition(".")) + sentence[index_match+1:]
                try:
                    sentence.remove("")
                except:
                    pass
        tokens += sentence
        length += len(sentence)
        boundary.add(length-1)
    tokens += "</s>"

    for i, token in enumerate(tokens):
        if i == len(tokens)-1:
            break
        if token in ".?!":
            if token == ".":
                punctuation = 0
            elif token == "?":
                punctuation = 1
            elif token == "!":
                punctuation = 2
            if tokens[i+1][0].isupper():
                is_next_word_capitalized = 1
            else:
                is_next_word_capitalized = 0
            if tokens[i-1][0].isdigit():
                is_prev_digit = 1
            else:
                is_prev_digit = 0
            if tokens[i-1]+tokens[i] in abbreviations:
                is_prev_abv = 1
            else:
                is_prev_abv = 0
            if i in boundary:
                row = np.array([punctuation, is_next_word_capitalized, is_prev_digit, is_prev_abv, "1"])
            else:
                row = np.array([punctuation, is_next_word_capitalized, is_prev_digit, is_prev_abv, "0"])
            features = np.append(features, [row], axis=0)
    return features


def prepareSplittingData(data):
    features = np.empty([0, 4])

    tokens = tokenize.word_tokenize(data)
    for t in tokens:
            if "." in t and len(t) > 1:
                index_match = tokens.index(t)
                tokens = tokens[:index_match] + list(t.partition(".")) + tokens[index_match+1:]
                while True:
                    try:
                        tokens.remove("")
                    except:
                        break
    tokens += "</s>"

    for i, token in enumerate(tokens):
        if i == len(tokens)-1:
            break
        if token in ".?!":
            if token == ".":
                punctuation = 0
            elif token == "?":
                punctuation = 1
            elif token == "!":
                punctuation = 2
            if tokens[i+1][0].isupper():
                is_next_word_capitalized = 1
            else:
                is_next_word_capitalized = 0
            if tokens[i-1][0].isdigit():
                is_prev_digit = 1
            else:
                is_prev_digit = 0
            if tokens[i-1]+tokens[i] in abbreviations:
                is_prev_abv = 1
            else:
                is_prev_abv = 0
            row = np.array([punctuation, is_next_word_capitalized, is_prev_digit, is_prev_abv])
            features = np.append(features, [row], axis=0)
    #print(features)
    return features

def trainModel(sentences):
    global LR

    colnames = ["punct","is_next_capitalized","is_prev_digit","is_prev_abv", "label"]
    featureSet = pd.DataFrame(sent_features(sentences), columns=colnames)

    x = featureSet[["punct","is_next_capitalized","is_prev_digit","is_prev_abv"]]
    y = featureSet.label

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    LR = LogisticRegression()
    LR.fit(x_train,y_train)

    #TEST THE MODEL
    
    accuracy_score = LR.score(x_test, y_test)
    print("ACCURACY:",accuracy_score)


def sent_split(data):
    
    colnames = ["punct","is_next_capitalized","is_prev_digit","is_prev_abv"]
    featureSet = pd.DataFrame(prepareSplittingData(data), columns=colnames)

    boundaries = LR.predict(featureSet)

    output = []
    current_char = 0

    boundaryPunct = re.compile("\.|\?|!")
    for i, m in enumerate(re.finditer(boundaryPunct, data)):
        if int(boundaries[i]) == 1:
            output.append(data[current_char:m.span()[1]].strip())
            current_char = m.span()[0] + 1
    return output


mydata = "Sagan, kitabında birkaç konuyu ele alıyor, bunlar temel olarak dünya dışı zeka olasılığına, daha gelişmiş uygarlıkların var olma olasılığına, bunların yerel galaksideki ve evrendeki dağılımına odaklanıyor. Daha gelişmiş zekaların varsayımsal görüşlerini, Dünya hakkındaki görüşlerini ve ayrıca insanlıkla iletişimini anlatıyor. Ayrıca UFO gözlemlerinin popülaritesini tartışıyor ve bu tür olayların olasılığını matematiksel olarak tasvir etmeye çalışıyor. Sagan ayrıca astrolojiyi bir sahte bilim olarak ele alıyor."


if __name__ == '__main__':
    trainModel(corpus)
    print(sent_split(mydata))

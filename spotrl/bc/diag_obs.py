"""Диагностика достаточности obs v2 для головы 0 (STAY/FLIP) клона v7.

Дешёвый регуляризованный тест (LR L2 + неглубокое дерево) на СЕМАНТИЧЕСКИХ
входах правила v7 (без сырых ценовых/объёмных осей = каналов мемоизации, без
самих решений v7 exit_sig/entry_signal). Две задачи раздельно: ВЫХОД (EXIT vs
HOLD на in-position барах) и ВХОД (ENTER vs WAIT на flat-барах). Train на одной
эпохе, тест на другой (обе стороны). Меряем AUC/AP (разделимость, НЕ L∞-зазор) и
recall@порог с печатью FPR рядом. ±pos_tag.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/home/cubecloud/Data/rlbinancetrader"

NOISE = ["m_ret_close","m_ret_high","m_ret_low","m_vwap_ret","m_hi_close",
         "m_close_lo","m_aggr_buy_frac","m_avg_trade_size","m_rel_volume"]
# семантические входы правила v7 (без решений v7 и без шумовых осей)
SEM_BASE = ["m_buy_margin","m_sell_margin","m_bounce_pct","m_leg_age","m_leg_dn",
            "m_regime_code","a_unreal_pnl","a_peak_unreal","a_price_drawdown",
            "a_dist_to_sl","a_dist_to_tp","a_entered_on_up","a_cb_active",
            "a_cb_cleared_today","a_cooldown_remain","a_days_in_trade",
            "w_breaker_armed","w_equity_drawdown"]
POSTAG = ["a_pos_tag_none","a_pos_tag_dip","a_pos_tag_transition"]
# решения v7 — исключаются из соответствующей задачи (циркулярно)
EXIT_DECISION = ["m_exit_flag"]
ENTRY_DECISION = ["m_entry_signal","m_trans_entry_signal"]

def load():
    d={}
    for ep in ("2021","2024"):
        d[ep]=pd.read_parquet(f"{DATA}/bc_clone_v7_{ep}.parquet")
    return d

def std(Xtr,Xte):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd=np.where(sd<1e-9,1.0,sd)
    return (Xtr-mu)/sd,(Xte-mu)/sd

def recall_at_train_thr(str_tr,ytr,str_te,yte,target_recall=0.995):
    # порог по train: минимальный t, дающий recall>=target на train позитивах
    pos=np.sort(str_tr[ytr==1])
    if len(pos)==0: return np.nan,np.nan
    k=int(np.ceil((1-target_recall)*len(pos)))
    thr=pos[min(k,len(pos)-1)]  # k-й снизу => recall≈target на train
    pred=str_te>=thr
    rec=pred[yte==1].mean() if (yte==1).any() else np.nan
    fpr=pred[yte==0].mean() if (yte==0).any() else np.nan
    return rec,fpr

def run_task(d, feats, mask_fn, pos_fn, name):
    print(f"\n===== {name} | признаков={len(feats)} =====")
    for tr,te in (("2024","2021"),("2021","2024")):
        dtr,dte=d[tr],d[te]
        mtr,mte=mask_fn(dtr),mask_fn(dte)
        Xtr=dtr.loc[mtr,feats].to_numpy(np.float64); ytr=pos_fn(dtr)[mtr].astype(int)
        Xte=dte.loc[mte,feats].to_numpy(np.float64); yte=pos_fn(dte)[mte].astype(int)
        Xtr_s,Xte_s=std(Xtr,Xte)
        # LR L2 (сильная регуляризация)
        lr=LogisticRegression(C=0.5,class_weight="balanced",max_iter=2000)
        lr.fit(Xtr_s,ytr)
        s_in=lr.decision_function(Xtr_s); s_out=lr.decision_function(Xte_s)
        auc_in=roc_auc_score(ytr,s_in); auc_out=roc_auc_score(yte,s_out)
        ap_in=average_precision_score(ytr,s_in); ap_out=average_precision_score(yte,s_out)
        rec,fpr=recall_at_train_thr(s_in,ytr,s_out,yte)
        rec_in,fpr_in=recall_at_train_thr(s_in,ytr,s_in,ytr)
        # неглубокое дерево
        dt=DecisionTreeClassifier(max_depth=4,class_weight="balanced",random_state=0)
        dt.fit(Xtr_s,ytr)
        p_in=dt.predict_proba(Xtr_s)[:,1]; p_out=dt.predict_proba(Xte_s)[:,1]
        tauc_in=roc_auc_score(ytr,p_in); tauc_out=roc_auc_score(yte,p_out)
        print(f" {tr}->{te}: pos_tr={ytr.sum()}/{len(ytr)} pos_te={yte.sum()}/{len(yte)}")
        print(f"   LR  AUC in={auc_in:.4f} out={auc_out:.4f} | AP in={ap_in:.4f} out={ap_out:.4f}"
              f" | recall in={rec_in:.3f}(fpr={fpr_in:.2e}) out={rec:.3f}(fpr={fpr:.2e})")
        print(f"   TREE AUC in={tauc_in:.4f} out={tauc_out:.4f}")

def main():
    d=load()
    inpos=lambda df:(df.a_in_position>0.5).to_numpy()
    flat =lambda df:(df.a_in_position<0.5).to_numpy()
    y_exit =lambda df:df.is_flip_exit.to_numpy()
    y_entry=lambda df:df.is_flip_entry.to_numpy()

    # --- ВЫХОД ---
    run_task(d, SEM_BASE, inpos, y_exit, "ВЫХОД: семантика БЕЗ pos_tag")
    run_task(d, SEM_BASE+POSTAG, inpos, y_exit, "ВЫХОД: семантика + pos_tag")
    run_task(d, SEM_BASE+POSTAG+NOISE, inpos, y_exit, "ВЫХОД: +шумовые оси (контраст мемоизации)")
    # --- ВХОД ---
    run_task(d, SEM_BASE, flat, y_entry, "ВХОД: семантика (pos_tag=none на flat)")
    run_task(d, SEM_BASE+NOISE, flat, y_entry, "ВХОД: +шумовые оси (контраст)")

    # --- Механический подтест: популяция exit_sig=1 в позиции ---
    print("\n===== ПОДТЕСТ: разделяет ли pos_tag+regime held-vs-exit при exit_sig=1 =====")
    for ep in ("2021","2024"):
        df=d[ep]; m=(df.a_in_position>0.5)&(df.m_exit_flag>0.5)
        held=int((m&~df.is_flip_exit).sum()); exd=int((m&df.is_flip_exit).sum())
        # доля held, объяснённая (trans & rc==2)
        expl=int((m&~df.is_flip_exit&(df.a_pos_tag_transition>0.5)&(df.m_regime_code>1.5)).sum())
        print(f" {ep}: exit_sig=1&in_pos: held={held} exited={exd}; "
              f"held объяснено (trans&bull)={expl}/{held}")

if __name__=="__main__":
    main()

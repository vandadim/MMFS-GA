function ACC_mtl = Acc_F1_Balance (Actual, Y_pred, N)

    Conf_Matrix    = confusionmat(Actual,Y_pred);
    TN             = Conf_Matrix(1,1);
    TP             = Conf_Matrix(2,2);
    FP             = Conf_Matrix(1,2);
    FN             = Conf_Matrix(2,1);
    Sensitivity       = TP / (TP + FN);
    Specificity       = TN / (TN + FP);
    Precision         = TP/(TP + FP);
    if N==1
        F1_SCORE= 2 * ((Precision * Sensitivity) / (Precision + Sensitivity));
        ACC_mtl=F1_SCORE;
    else
        Balanced_Accuracy = (Sensitivity + Specificity) / 2;
        ACC_mtl=Balanced_Accuracy;
    end
    
    
end

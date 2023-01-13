class Encode_Transformer():
    """
    This transformer transforms 4 categorical values to numerical values, including
    person_home_ownership, loan_grade, loan_intent, and cb_person_default_on_file.
    The first set of ordinal encoder number is determined randomly;
    another ordinal encoder number is determined quantitatively (@Ming Gu);
    the third transformation encoder is used now (@Zulin Yu)
    (can test other methods later).
    """
    def fit(self, X, y=None):
        return
    
    def transform(self, X, y=None):
        df = X     
        df['person_home_ownership'] = df.person_home_ownership.map({'MORTGAGE':2, 'OTHER':0, 'OWN':3, 'RENT':1})
        df['loan_grade'] = df.loan_grade.map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6})
        df['loan_intent'] = df.loan_intent.map({'DEBTCONSOLIDATION':0, 'EDUCATION':1, 'HOMEIMPROVEMENT':2,
                                                          'MEDICAL':3, 'PERSONAL': 4, 'VENTURE':5})
        df['cb_person_default_on_file'] = df.cb_person_default_on_file.map({'Y':1, 'N':0})
        
        return df
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)
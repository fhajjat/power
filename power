 
 class FactorialDesigns:
  

  def __init__(self, input_data, factor_cols, response_var, covariates = None, effect_size = 0.1, alpha = 0.05, bootstrap_num = 5):

    '''
    Attributes:
    input_data(pandas df): the data to bootstrap and use to calculate power
    factor_cols (list): the list of factor column names (i.e., the independent variable(s))
    response_var (str) : the dependent variable 
    covariates (list, optional, default = None): the list of control variables 
    effect_size (int): the effect size to use for calculating power

    '''
    self.input_data = input_data
    self.factor_cols = factor_cols
    self.response_var = response_var
    self.effect_size = effect_size
    self.alpha = alpha
    self.bootstrap_num = bootstrap_num
    self.covariates = covariates

  def calculate_fstat (self, data):

    response_var_arr = np.array(data[self.response_var])
    if self.covariates:
      formula = f"{self.response_var} ~ {'+ '.join(self.factor_cols + self.covariates)}"
      model = ols(formula, data = data).fit()
      data['fitted'] = model.fittedvalues
      group_means = data.groupby(self.factor_cols)['fitted'].mean(axis =0)
    else:
      group_means = data.groupby(self.factor_cols)[self.response_var].mean()

    group_sizes = data.groupby(self.factor_cols).size()
  
    #Calculate the ss and df for each factor
    ss_between ={}
    df_between ={}
    for factor in self.factor_cols:
      for level in data[factor].unique():
        subgroup = data[data[factor] == level]
        if self.covariates:
          group_mean = subgroup['fitted'].mean()
        else:
          group_mean = subgroup[self.response_var].mean()
        ss_between [factor, level] = len(subgroup) * (group_mean - np.mean(data[self.response_var])) **2
        df_between [factor, level] = 1

    total_ss_between = sum(ss_between.values())
    total_df_between = sum(df_between.values())

    #calculate the total SS and the DF
    #calculate the within SS and DF

    if self.covariates:
      total_ss = sum((data['fitted'] - np.mean(data[self.response_var]))**2)
      ss_within = sum((data[self.response_var] - data['fitted'])**2)
    else:
      total_ss = sum([((x-np.mean(response_var_arr))**2) for x in response_var_arr])
      ss_within = total_ss - total_ss_between

    total_df = data.shape[0]-1
    df_within = total_df - total_df_between

    #Calculate the mean squares 
    ms_between = total_ss_between/ total_df_between 
    ms_within = ss_within/df_within

    #calculate the f-statistic
    f_stat = ms_between/ms_within
    p_value = 1- stats.f.cdf(f_stat, total_df_between, df_within)
    return f_stat, p_value, total_df_between, df_within, total_df, group_means, group_sizes

  def calculate_power(self):

    f_stat, p_value, total_df_between, df_within, total_df, group_means, group_sizes = self.calculate_fstat(data= self.input_data)

    count_sig_pvalue =0
    count_sig_fstat = 0
    for i in range (self.bootstrap_num):
      bootstrapped_data = self.input_data.sample(frac = 1, replace = True)
      f_stat, p_value, df_between, df_within, _, _, _ = self.calculate_fstat(data= bootstrapped_data)

      critical_f = stats.f.ppf(1-self.alpha, df_between, df_within)

      # print (f_stat, p_value, critical_f)
      if (p_value < self.alpha).all():
        pvalue_significant+=1
      
      if (f_stat > critical_f).all():
        fstat_larger+=1

      pvalue_significant = count_sig_pvalue/self.bootstrap_num
      fstat_larger = count_sig_fstat/self.bootstrap_num

    return pvalue_significant, fstat_larger

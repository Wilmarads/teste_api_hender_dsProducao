import pickle
import inflection
import pandas as pd
import numpy  as np
import math
import datetime

class Rossmann( object ):
  '''A classe Rossmann é um conjunto de métodos para manipulação, limpeza e preparação de dados para um modelo preditivo, além de realizar previsões. 
  Seus métodos incluem desde a limpeza e transformação dos dados até a preparação final para a previsão. 
  Isso envolve ajustes nos tipos de dados, criação de novas features, escalonamento, codificação de variáveis categóricas e geração de previsões a 
  partir de um modelo treinado.

  Input: 
  1. df1, df2, df5: DataFrames com dados para limpeza, engenharia de features e preparação, respectivamente.
  2. model: Modelo de machine learning treinado.
  3. original_data: Dados originais para os quais as previsões serão incorporadas.
  4. test_data: Dados de teste para gerar previsões.

  Output:

  Métodos de limpeza, engenharia de features e preparação retornam DataFrames processados.
  get_prediction() retorna os dados originais com uma nova coluna contendo as previsões do modelo em formato JSON.

  Esses métodos encapsulam etapas comuns em pipelines de pré-processamento e previsão de modelos de machine learning aplicados ao contexto 
  específico da previsão de vendas da Rossmann.
  '''
  def __init__( self ):
    '''Esta função __init__ da classe Rossmann é um método especial que é chamado quando uma instância da classe é criada. 
    Ela é responsável por inicializar os atributos da classe, configurando os caminhos dos arquivos e carregando os scalers necessários para o 
    pré-processamento dos dados no contexto do modelo de previsão de vendas da Rossmann.

    Input: Não recebe input direto, mas depende dos arquivos presentes no caminho definido em self.home_path.
    Output:

    Atributos da classe Rossmann são inicializados, contendo os scalers carregados a partir dos arquivos específicos definidos no caminho self.home_path. 
    Estes scalers são utilizados em etapas posteriores de pré-processamento e transformação dos dados para o modelo de previsão.
    '''
    self.home_path=''
    self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
    self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
    self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
    self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
    self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )

  def data_cleaning( self, df1 ):
    '''A função data_cleaning na classe Rossmann realiza diversas operações de pré-processamento nos dados para padronizá-los e prepará-los para análise.
    Isso inclui a renomeação de colunas para o formato snake_case, conversão de tipos de dados (como a coluna 'date' para datetime), preenchimento de 
    valores ausentes em algumas colunas específicas ('competition_distance', 'competition_open_since_month', 'competition_open_since_year', 
    'promo2_since_week', 'promo2_since_year', 'promo_interval'), e criação de novas features com base nas colunas existentes.

    Input: Recebe um DataFrame df1 contendo os dados a serem limpos e preparados.

    Output: Retorna o DataFrame df1 após a aplicação das transformações de limpeza e pré-processamento. 
    Este DataFrame estará pronto para ser usado em etapas subsequentes, como engenharia de features ou modelagem. 
    '''
    ## 1.1. Rename Columns
    cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    
    snakecase = lambda x: inflection.underscore( x )

    cols_new = list( map( snakecase, cols_old ) )
    
    # rename
    df1.columns = cols_new
    
    ## 1.3. Data Types
    df1['date'] = pd.to_datetime( df1['date'] )
    
    ## 1.5. Fillout NA
    
    #competition_distance
    df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )
    
    #competition_open_since_month
    df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )

    #competition_open_since_year
    df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )

    #promo2_since_week
    df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )

    #promo2_since_year
    df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )

    #promo_interval
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6:'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    df1['promo_interval'].fillna(0, inplace=True )

    df1['month_map'] = df1['date'].dt.month.map( month_map )

    df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )

    ## 1.6. Change Data Types
    # competiton
    df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
    df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int )

    # promo2
    df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
    df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
    return df1

  def feature_engineering( self, df2 ):
    '''A função feature_engineering na classe Rossmann realiza a criação e transformação de features nos dados, adicionando informações relevantes que 
    podem ajudar no processo de modelagem. Esta função inclui a extração de informações temporais (como ano, mês, dia, semana do ano), cálculos de tempo
    desde eventos específicos (como a competição desde a abertura da loja), e transformações de variáveis categóricas para representações numéricas mais 
    úteis.

    Input: Recebe um DataFrame df2 contendo os dados a serem processados e enriquecidos com novas features.

    Output: Retorna o DataFrame df2 após a aplicação das transformações de engenharia de features. Este DataFrame terá colunas adicionais que capturam 
    informações temporais, calculadas a partir dos dados existentes, e outras features derivadas para melhorar o desempenho do modelo de previsão.
    '''
    # year
    df2['year'] = df2['date'].dt.year

    # month
    df2['month'] = df2['date'].dt.month

    # day
    df2['day'] = df2['date'].dt.day

    # week of year
    df2['week_of_year'] = df2['date'].dt.isocalendar().week

    # year week
    df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )
    
    # competition since
    df2['competition_since'] = df2.apply( lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
    df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

    # promo since
    df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
    df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
    df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

    # assortment
    df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x =='a' else 'extra' if x == 'b' else 'extended' )

    # state holiday
    df2['state_holiday'] = df2['state_holiday'].apply( lambda x:'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

    # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
    ## 3.1. Filtragem das Linhas
    df2 = df2[df2['open'] != 0]

    ## 3.2. Selecao das Colunas
    cols_drop = ['open', 'promo_interval', 'month_map']
    df2 = df2.drop( cols_drop, axis=1 )

    return df2

  def data_preparation( self, df5 ):
    '''A função data_preparation na classe Rossmann executa o processo final de preparação dos dados, realizando escalonamento de variáveis numéricas, 
    codificação de variáveis categóricas e criação de novas features para serem utilizadas no modelo de previsão. 
    Essa etapa é fundamental para garantir que os dados estejam formatados corretamente e prontos para serem usados no treinamento do modelo.

    Input: Recebe um DataFrame df5 que contém os dados após passarem por etapas anteriores de limpeza e engenharia de features.

    Output: Retorna um DataFrame com as colunas selecionadas e as transformações finais aplicadas, pronto para ser utilizado no treinamento do modelo 
    de machine learning. Essas transformações incluem escalonamento de variáveis numéricas, codificação de variáveis categóricas e a geração de features
    adicionais para enriquecer os dados.
    '''
    ## 5.2. Rescaling
    # competition distance
    df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )
    
    # competition time month
    df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values )
    
    # promo time week
    df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values )
    
    # year
    df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )

    ### 5.3.1. Encoding
    # state_holiday - One Hot Encoding
    df5 = pd.get_dummies( df5, prefix=['state_holiday'],columns=['state_holiday'] )

    # store_type - Label Encoding
    df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'] )

    # assortment - Ordinal Encoding
    assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
    df5['assortment'] = df5['assortment'].map( assortment_dict )

    ### 5.3.3. Nature Transformation
    # day of week
    df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
    df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

    # month
    df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
    df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

    # day
    df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )
    
    # week of year
    df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin(x * ( 2. * np.pi/52 ) ) )
    df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos(x * ( 2. * np.pi/52 ) ) )

    cols_selected = [ 'store', 'promo', 'store_type', 'assortment','competition_distance', 'competition_open_since_month',
                      'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month',                               'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos','day_sin', 'day_cos',                                   'week_of_year_sin', 'week_of_year_cos']
    
    return df5[ cols_selected ]

  def get_prediction( self, model, original_data, test_data ):
    '''A função get_prediction na classe Rossmann é responsável por gerar previsões usando um modelo treinado e incorporar essas previsões aos dados 
    originais. Ela recebe o modelo treinado, os dados originais e os dados de teste, executa a previsão usando o modelo nos dados de teste e junta essas 
    previsões ao conjunto de dados original.

    Input:

    1. model: Modelo de machine learning previamente treinado.
    2. original_data: Dados originais aos quais as previsões serão adicionadas.
    3. test_data: Dados de teste nos quais o modelo fará previsões.

    Output: Retorna um JSON contendo os dados originais com uma nova coluna chamada 'prediction', que contém as previsões feitas pelo modelo para os dados 
    de teste. Este formato permite uma fácil visualização das previsões associadas aos dados originais. Esses dados estão prontos para serem analisados ou 
    utilizados em outras aplicações.
    '''
    # prediction
    pred = model.predict( test_data )

    # join pred into the original data
    original_data['prediction'] = np.expm1( pred )
    
    return original_data.to_json( orient='records', date_format='iso' )


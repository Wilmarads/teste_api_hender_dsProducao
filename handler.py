import os
import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
current_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório atual do script
model_path = os.path.join(current_dir, 'model', 'model_rossmann.pkl')

# Carregando o modelo
with open(model_path, 'rb') as file:
    model = pickle.load(file)
# model = pickle.load( open( '/model/model_rossmann.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    '''A função rossmann_predict é uma função de roteamento em uma aplicação web usando o framework Flask que processa solicitações de previsão para 
    um modelo Rossmann. Ela recebe dados de teste em formato JSON, limpa, executa engenharia de features, prepara os dados e realiza previsões usando 
    um modelo previamente treinado da classe Rossmann.

    Input:

    1. Dados de teste em formato JSON, recebidos por meio de uma solicitação web (solicitação POST).
    2. model: Modelo de machine learning previamente treinado.

    Output:

    1. Retorna um JSON contendo os dados originais com uma nova coluna chamada 'prediction', que contém as previsões feitas pelo modelo para os dados de teste.
    2. Caso não haja dados de teste, retorna uma resposta vazia em formato JSON com status 200.
    
    A função é projetada para ser integrada a um servidor web, onde a rota '/rossmann_predict' manipula as solicitações de previsão enviadas para esse
    endpoint específico.
    '''
    test_json = request.get_json()

    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )

        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

        # Instantiate Rossmann class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )

        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        # data preparation

        df3 = pipeline.data_preparation( df2 )

        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )

        return df_response

    else:
        return Reponse( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )



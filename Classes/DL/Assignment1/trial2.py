dc = {'activation_functions': 'relu', 'batch_size': 64, 'learning_rate': 0.0001, 'number_of_epochs': 5, 'number_of_hidden_layers': 3,
      'optimizer': 'sgd', 'size_of_every_hidden_layer': 32, 'weight_decay': 0.0005, 'weight_initialisation': 'xavier_uniform'}


def get_name(params):
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]
    
    name = ''
    for key,val in zip(keys,values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name

print(get_name(dc))

import study as s


def run():
    import numpy as np
    import autokeras as ak
    from sklearn.model_selection import train_test_split
    import pickle
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.models import model_from_json

    data  = s.open_data()
    score = s.open_score()

    print('pn, c1, c2, c3.\nSelect!')
    flag = True
    while flag:
        print('study_type = ', end='', flush=True)
        study_type = input()
        if study_type == 'pn':
            flag = False
        elif study_type == 'c1':
            data = s.c1(data)
            flag = False
        elif study_type == 'c2':
            data = s.c2(data)
            flag = False
        elif study_type == 'c3':
            data = s.c3(data)
            flag = False
        else:
            flag = True

    score = score / 100

    x_train,x_test,y_train,y_test = train_test_split(data, score, test_size=0.5)

    reg = ak.StructuredDataRegressor(max_trials=3)
    reg.fit(x_train, y_train, epochs=5)
    eva = reg.evaluate(x_test, y_test)

    eva_name = './result/' + study_type + '_3.txt'
    np.savetxt(eva_name, eva)

    model = reg.export_model()

#    json_string = model.to_json()
#    json_name = './result/' + study_type + '_3.json'
#    with open(json_name, 'w', encoding='utf-8') as f:
#        f.write(json_string)

    model_name = './result/' + study_type + '_3.h5'
    model.save(model_name, save_format='tf')

    pdf_name = './result/' + study_type + '_3.pdf'
    plot_model(model, to_file=pdf_name)

    # weights_name = study_type + '.hdf5'
    # model.save_weights(weights_name)

    return model, x_test, y_test

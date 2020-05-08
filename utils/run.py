from utils.model import Model


def run():
    answ = int(input('Predict or Train?(1/2): '))

    if answ == 1:
        person = input('Enter the number of person(1-16): ')
        predict = input('Predict: ')

        result = Model.predict(person, predict)

        for key in result:
            print(key, ':', '\n', result[key])

        return run()

    elif answ == 2:
        person = input('Enter the number of person(01-16): ')
        predict = input('Predict: ')

        Model.prepare_model(person, predict)

        return run()

    else:
        print('Bye!')
run()

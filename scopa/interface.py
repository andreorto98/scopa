import argparse
import cv2
import sys

from scopa.utilities import get_area, import_deck, Match

optimal_area = 21500

# export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is a program that can play 'scopa l'asso'.\
                                     Launch this script from /scopa/scopa directory and remember \
                                     to correctly set your PYTHONPATH",
                                     epilog='Developed by Andrea Ortone.')
    parser.add_argument('url', type = str, help='Web address (Uniform Resource Locator) \
                        of the desidered IPcamera.')
    parser.add_argument('deck', nargs = '?', type = str, default = 'default',
                        help='Defines which deck you want to use: default or new; \
                        if new is passed the import_deck and training phases must be repeated. \
                        Default to default.')
    args = parser.parse_args()
    url = args.url
    default_deck = True if args.deck == 'default' else False

    #input_area= 21500

    print('Please, Insert a card')
    input_area = get_area(url)
    print(f'Detected area: {input_area}')

    while input_area < optimal_area*0.93 or input_area > optimal_area*1.07:
        print(f'Your_card_areas/Desidered_card_areas = {input_area/optimal_area}\t \
              Must be in [{optimal_area*0.95}, {optimal_area*1.05}]')
        input_area = get_area(url)

    if not default_deck:
        try:
            inp = int(input('Start to import new_deck from card: (1 to 40) [1] (if already imported type >40): '))
        except ValueError:
            inp = 1
        import_deck('./new_deck', url, start = inp)
        print('new_deck has been correctly imported.\n \
               Run the notebooks to save the models in ./new_models')
        path_to_models = './new_models'
        inp = input('Proceed? (y,n): ')
        if inp != 'y':
            sys.exit()
    else:
        path_to_models = './models'
                # check this area.... and get_area cards in different positions (21800 al centro, 21280 al bordo) ok 0.3 %


    print('End of setup operations. We can start playing.')

    mat = Match()
    print('path_to_model: '+path_to_models)
    mat(url, min_area = 0.7*input_area, margin = 25, path_to_models=path_to_models)

    # ora devi importare il nuovo mazzo e vedere che funzioni, costruire di nuovo i modelli, salvare i notebook  vedere se funziona, fare i test, scrivere un breve report e rivedere la documentation


    # controlla last_take fai printare i deck alla fine, vedi errore che ti da !! e transform img args inutile

from curtsies import Input

def main():
    while(True):
        with Input(keynames='curses') as input_generator:
            e = input_generator.send(0.1)
            print(input_generator)
            print(e=='1')
            print(repr(e))

if __name__ == '__main__':
    main()
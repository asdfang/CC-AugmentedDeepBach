import click
import music21

@click.command()
@click.argument('filename')

def main(filename):
  chorale = music21.converter.parse(filename)
  chorale.show('txt')
  
  grade(chorale)

def grade(chorale):
  return 0

if __name__ == '__main__':
    main()
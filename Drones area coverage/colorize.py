# Code inspired by
# - https://stackabuse.com/how-to-print-colored-text-in-python/
# - https://github.com/openai/gym/blob/master/gym/utils/colorize.py

#  Dict of colors
color2num = dict(
    gray=40,
    red=41,
    green=42,
    yellow=43,
    blue=44,
    magenta=45,
    cyan=46,
    white=47,
)

#  Highlight the backgournd of text. It's used to display a cover of a drone
#  Input: text which need a colored background (String), a color (String) and use or not of different colors (Boolean)
#  Output: Text with colored background (String)
def colorize(text, color_, small=True):
    if small:
        num = color2num[color_]
        return (f"\x1b[{num}m{text}\x1b[0m")
    else:
        num = color_
        return (f"\x1b[48;5;{num}m{text}\x1b[0m")

#  Color a text in yellow. Function only used for 30x30 map. It highlights drones with a perfect cover.
#  Input: text to highlight (String)
#  Output: highlighted text (String)
def yellow(text):
    return (f"\x1b[1;33m{text}\x1b[0m")





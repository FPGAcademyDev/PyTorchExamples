"""Interactive prompts for board size and winning rules."""

from neutreeko.io_utils import read_int
from neutreeko.models import GameConfig


def prompt_game_config() -> GameConfig:
    print("\nNeutreeko (c) 2001, 2002 J K Haugland")
    print("\nEnter 0 for standard Neutreeko")
    print("\nOtherwise, enter board size:")

    width = 0
    height = 0
    ruleset = 0

    while width < 3 or width > 7:
        width = read_int("Width = ")
        if width == 0:
            width = 5
            height = 5
            ruleset = 1
            print("\nWidth = 5")
            print("Height = 5")
            print("Winning condition: three in a row, orthogonally or diagonally")
        if width < 3:
            print("Too narrow")
        if width > 7:
            print("Too wide")

    while height < 3 or height > 7:
        height = read_int("Height = ")
        if height < 3:
            print("Too low")
        if height > 7:
            print("Too high")

    if ruleset == 0:
        print("")

    w, h = width, height
    while (
        ruleset < 1
        or ruleset > 5
        or (ruleset == 3 and (w == 3 or h == 3))
        or (ruleset == 4 and w < 5 and h < 5)
        or (ruleset == 5 and (w % 2 == 0 or h % 2 == 0 or w * h < 25))
    ):
        print("Enter winning condition:")
        print("1 = three in a row, orthogonally or diagonally")
        print("2 = three in a row, orthogonally")
        if w > 3 and h > 3:
            print(
                "3 = three in a row, diagonally"
            )  # width or height = 3 would make it possible to be trapped with no legal moves
        if w > 4 or h > 4:
            print(
                "4 = three in a straight line, any equidistant"
            )  # either width or height must be greater than 3 to make this distinct from no. 1
        if (w == 5 or w == 7) and (h == 5 or h == 7):
            print(
                "5 = occupy the centre"
            )  # only possible when width and height are odd - also cf. no. 3
        ruleset = read_int("")

    return GameConfig(width=width, height=height, ruleset=ruleset)

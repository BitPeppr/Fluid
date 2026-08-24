gradient = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. '[::-1]
gradient_len = len(gradient)


def convert_ascii(value, array_max, array_min):
    value_range = array_max - array_min
    if value_range == 0:
        return gradient[0]
    normalised = (value - array_min) / value_range
    return gradient[min(int(normalised * gradient_len), gradient_len - 1)]

def render(array):
    array_max = array.max()
    array_min = array.min()

    ascii_array = []
    for row in array[::-1]:
        ascii_row = []
        for value in row:
            ascii_value = convert_ascii(value, array_max, array_min)
            ascii_row.append("".join(ascii_value))
        ascii_array.append(''.join(ascii_row))
    return '\n'.join(ascii_array)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def change_brightness(image, value):
    hsv = HSV(image)
    h, s, v = cv2.split(hsv)
    v = cv2.addWeighted(src1=v, alpha=value, src2=0, beta=0, gamma=0)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def change_saturation(image, value):
    hsv = HSV(image)
    h, s, v = cv2.split(hsv)
    s = cv2.addWeighted(src1=s, alpha=value, src2=0, beta=0, gamma=0)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def warm_image(image, value):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image)
    r = cv2.addWeighted(src1=r, alpha=value, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=value, src2=0, beta=0, gamma=0)
    final_rgb = cv2.merge((r, g, b))
    return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)


def cold_image(image, value):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image)
    b = cv2.addWeighted(src1=b, alpha=value, src2=0, beta=0, gamma=0)
    final_rgb = cv2.merge((r, g, b))
    return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)


def binarise_image(image, threshold):
    grayscale_image = grayscale(image)
    th, dst = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)
    print(th)
    return dst


def draw_contours(image, threshold):
    binary_img = binarise_image(image, threshold)
    contours, ier = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = cv2.drawContours(image, contours, -1, (0, 0, 255), 8)
    return result_img


def shift_image(image, x, y):
    shift = np.array([[1, 0, x],
                      [0, 1, y]]).astype(np.float32)
    return cv2.warpAffine(image, shift, dsize=(image.shape[1], image.shape[0]))


def rotate_image(image, angle, center):
    rows, cols, _ = image.shape
    R = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, R, (cols, rows))


image = cv2.imread('yellow-smile.jpg', cv2.IMREAD_COLOR)
cv2.imwrite('rorate-smile.jpg', rotate_image(image, 45,
            (image.shape[1] / 2.0, image.shape[0] / 2.0)))
cv2.imwrite('shift-smile.jpg', shift_image(image, -100, 0))
cv2.imwrite('contours-smile.jpg', draw_contours(image, 128))
cv2.imwrite('binarise-smile.jpg', binarise_image(image, 128))
cv2.imwrite('cold-smile.jpg', cold_image(image, 1.2))
cv2.imwrite('warm-smile.jpg', warm_image(image, 1.2))
cv2.imwrite('saturation-smile.jpg', change_saturation(image, 3))
cv2.imwrite('brightness-smile.jpg', change_brightness(image, 1.5))
cv2.imwrite('hsv-smile.jpg', HSV(image))
cv2.imwrite('gray-smile.jpg', grayscale(image))

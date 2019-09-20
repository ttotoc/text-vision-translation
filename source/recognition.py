import cv2
import pytesseract


def perform(image, boxes=None):
    # cli arguments
    from arguments import ARGS

    # tesseract config
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 1, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the image as a single line of text
    config = ARGS.tesseract_config

    # perform recognition on whole image if no boxes
    if boxes is None:
        return [pytesseract.image_to_string(image, config=config)]

    # list of results
    results = []

    # loop over the bounding boxes
    for (start_x, start_y, end_x, end_y) in boxes:

        # extract the actual padded ROI
        roi = image[start_y:end_y, start_x:end_x]

        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((start_x, start_y, end_x, end_y), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])  # key = start_y

    # make a copy the output image
    output = image.copy()

    # loop over the results and show them on the image
    for ((start_x, start_y, end_x, end_y), text) in results:
        # display the text OCR'd by Tesseract
        print(f'Recognized text: "{text}"')

        # remove the non-ASCII characters from the resulting text
        text = "".join((c if ord(c) < 128 else "" for c in text)).strip()

        # draw the text and a bounding box surrounding the text region of the input image
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), thickness=2)
        cv2.putText(output, text, (start_x, start_y - 10),
                     cv2.FONT_HERSHEY_PLAIN, fontScale=1.1, color=(50, 50, 255), thickness=2)

    cv2.imshow(ARGS.image, output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # return text results
    if ARGS.concatenate:
        return [' '.join([result[1] for result in results])]

    return [result[1] for result in results]

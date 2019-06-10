import cv2
import pytesseract


def perform(image, boxes):
    # cli arguments
    from arguments import ARGS

    orig_height, orig_width = image.shape[:2]

    # determine the ratio change for both the width and height
    ratio_width = orig_width / float(ARGS.width)
    ratio_height = orig_height / float(ARGS.height)

    # list of results
    results = []

    # loop over the bounding boxes
    for (start_x, start_y, end_x, end_y) in boxes:
        # scale the bounding box coordinates based on the respective ratios
        start_x = int(start_x * ratio_width)
        start_y = int(start_y * ratio_height)
        end_x = int(end_x * ratio_width)
        end_y = int(end_y * ratio_height)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        delta_x = int((end_x - start_x) * ARGS.padding)
        delta_y = int((end_y - start_y) * ARGS.padding)

        # apply padding to each side of the bounding box, respectively
        start_x = max(0, start_x - delta_x)
        start_y = max(0, start_y - delta_y)
        end_x = min(orig_width, end_x + (delta_x * 2))
        end_y = min(orig_height, end_y + (delta_y * 2))

        # extract the actual padded ROI
        roi = image[start_y:end_y, start_x:end_x]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = "-l eng --oem 1 --psm 7"
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
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), thickness=1)
        cv2.putText(output, text, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1.6, color=(50, 50, 255), thickness=1)

    cv2.imshow(ARGS.image, output)

    # freeze until keypress
    cv2.waitKey(0)

    # return text results
    if ARGS.concatenate:
        return [' '.join([result[1] for result in results])]

    return [result[1] for result in results]

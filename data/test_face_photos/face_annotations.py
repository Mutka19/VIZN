annotations = [
    {'photo_file_name': 'clintonAD2505_468x448.jpg', 'faces': [[146, 226, 96, 176], [56, 138, 237, 312]]},
    {'photo_file_name': 'DSC01181.JPG', 'faces': [[141, 181, 157, 196], [144, 184, 231, 269]]},
    {'photo_file_name': 'DSC01418.JPG', 'faces': [[122, 147, 263, 285], [129, 151, 305, 328]]},
    {'photo_file_name': 'DSC02950.JPG', 'faces': [[126, 239, 398, 501]]},
    {'photo_file_name': 'DSC03292.JPG', 'faces': [[92, 177, 169, 259], [122, 200, 321, 402]]},
    {'photo_file_name': 'DSC03318.JPG', 'faces': [[188, 246, 178, 238], [157, 237, 333, 414]]},
    {'photo_file_name': 'DSC03457.JPG', 'faces': [[143, 174, 127, 157], [91, 120, 177, 206], [94, 129, 223, 257]]},
    {'photo_file_name': 'DSC04545.JPG', 'faces': [[56, 86, 119, 151]]},
    {'photo_file_name': 'DSC04546.JPG', 'faces': [[105, 137, 193, 226]]},
    {'photo_file_name': 'DSC06590.JPG', 'faces': [[167, 212, 118, 158], [191, 228, 371, 407]]},
    {'photo_file_name': 'DSC06591.JPG', 'faces': [[180, 222, 290, 330], [260, 313, 345, 395]]},
    {'photo_file_name': 'IMG_3793.JPG', 'faces': [[172, 244, 135, 202], [198, 253, 275, 331], [207, 264, 349, 410], [160, 233, 452, 517]]},
    {'photo_file_name': 'IMG_3794.JPG', 'faces': [[169, 211, 109, 148], [154, 189, 201, 235], [176, 204, 314, 342], [170, 206, 445, 483], [144, 191, 550, 592]]},
    {'photo_file_name': 'IMG_3840.JPG', 'faces': [[200, 268, 150, 212], [202, 262, 261, 323], [222, 286, 371, 430], [154, 237, 477, 549]]},
    {'photo_file_name': 'jackie-yao-ming.jpg', 'faces': [[45, 77, 93, 124], [61, 91, 173, 200]]},
    {'photo_file_name': 'katie-holmes-tom-cruise.jpg', 'faces': [[55, 102, 93, 141], [72, 116, 197, 241]]},
    {'photo_file_name': 'mccain-palin-hairspray-horror.jpg', 'faces': [[58, 139, 100, 179], [102, 177, 254, 331]]},
    {'photo_file_name': 'obama8.jpg', 'faces': [[85, 157, 109, 180]]},
    {'photo_file_name': 'phil-jackson-and-michael-jordan.jpg', 'faces': [[34, 75, 58, 92], [32, 75, 152, 193]]},
    {'photo_file_name': 'the-lord-of-the-rings_poster.jpg', 'faces': [[222, 267, 0, 35], [129, 170, 6, 40], [13, 81, 26, 84], [22, 92, 120, 188], [35, 94, 225, 276], [190, 255, 235, 289], [301, 345, 257, 298]]}
]

# Iterate over the annotations
for annotation in annotations:
    photo_file_name = annotation['photo_file_name']
    faces = annotation['faces']

    print(f"Processing {photo_file_name} with {len(faces)} faces")

    # Process each face
    for face in faces:
        top, bottom, left, right = face
        # Process the face
        # For example, you can print the coordinates
        print(f"Face coordinates: Top={top}, Bottom={bottom}, Left={left}, Right={right}")

        # Here you can add code to handle each face, such as drawing bounding boxes,
        # cropping the face region from the image, etc.


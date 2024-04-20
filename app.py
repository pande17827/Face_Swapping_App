import streamlit as st
import cv2
import numpy as np
import dlib
from PIL import Image
import os

# Function to perform face swapping
def face_swap(image1, image2, predictor_path):
    try:
        # Convert PIL images to numpy arrays
        img = np.array(image1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
        img2 = np.array(image2)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize frontal face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        height, width, channels = img2.shape
        img2_new_face = np.zeros((height, width, channels), np.uint8)

        # Face 1
        faces = detector(img_gray)
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, convexhull, 255)

            face_image_1 = cv2.bitwise_and(img, img, mask=mask)

            # Delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = landmarks_points.index(pt1)
                index_pt2 = landmarks_points.index(pt2)
                index_pt3 = landmarks_points.index(pt3)

                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    indexes_triangles.append(triangle)

        # Face 2
        faces2 = detector(img2_gray)
        for face in faces2:
            landmarks = predictor(img2_gray, face)
            landmarks_points2 = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)

        # Creating empty mask
        lines_space_mask = np.zeros_like(img_gray)
        lines_space_new_face = np.zeros_like(img2)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Lines space
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
            cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
            lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        # Creating seamless clone of two faces
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        # Return the swapped image
        return seamlessclone
    except Exception as e:
        st.error(f"An error occurred during face swapping: {e}")
        return None

# Streamlit app
st.set_page_config(page_title="Face Swapping App", page_icon=":camera:")

# Title and instructions
st.title("Face Swapping App")
st.markdown(
    """
    Upload two images and click the button to swap their faces!
    """
)

# Upload images
col1, col2 = st.columns(2)
with col1:
    image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
with col2:
    image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

# Check if predictor file exists in the current directory
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    st.error("Error: Predictor file not found. Please make sure 'shape_predictor_68_face_landmarks.dat' is in the current directory.")
else:
    # Perform face swapping on button click
    if image1 and image2:
        st.header("Original Images")
        col1.image(image1, caption="Image 1", use_column_width=True)
        col2.image(image2, caption="Image 2", use_column_width=True)

        if st.button("Swap Faces", key="swap_button"):
            st.header("Swapped Image")
            # Convert uploaded images to PIL images
            img1_pil = Image.open(image1)
            img2_pil = Image.open(image2)

            # Perform face swapping
            swapped_image = face_swap(img1_pil, img2_pil, predictor_path)

            # Display swapped image
            if swapped_image is not None:
                st.image(swapped_image, caption="Swapped Image", use_column_width=True)

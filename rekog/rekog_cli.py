import boto3
import string
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import uuid 
import json
import os
import re
import time
import logging


COLL_ID = os.getenv("ARH_COLL_ID", "default")
SIMILARITY_THRESHOLD = 99.9
COMP_FACE_THRESHOLD = 99.9

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arh")
logger.setLevel(logging.DEBUG)

rek_cli = boto3.client("rekognition")


def annotate(img, draw, point, text, font):
    w, h = font.getsize(text)
    
    # width, height = img.size
    xx, yy = (point[0]+w, point[1]+h)

    draw.rectangle((point[0], point[1], xx, yy), fill='black')    
    draw.text(point, text, fill='white', font=font)
    #draw.text((xx, yy), text, fill=(0, 0, 0, 15), font=font) # draw transparant text

    return draw


# Calculate positions from from estimated rotation 
def show_bounding_box_positions(image_height, image_width, box, rotation):
    logger.info(f">show_bounding_box_positions ih: {show_bounding_box_positions} iw: {image_width} l: {box}")
    left = 0
    top = 0
      
    if rotation == 'ROTATE_0':
        left = image_width * box['Left']
        top = image_height * box['Top']
    
    if rotation == 'ROTATE_90':
        left = image_height * (1 - (box['Top'] + box['Height']))
        top = image_width * box['Left']

    if rotation == 'ROTATE_180':
        left = image_width - (image_width * (box['Left'] + box['Width']))
        top = image_height * (1 - (box['Top'] + box['Height']))

    if rotation == 'ROTATE_270':
        left = image_height * box['Top']
        top = image_width * (1 - box['Left'] - box['Width'] )

    logger.info("       Left: {0:.0f}".format(left))
    logger.info("        Top: {0:.0f}".format(top))
    logger.info(" Face Width: {0:.0f}".format(image_width * box['Width']))
    logger.info("Face Height: {0:.0f}".format(image_height * box['Height']))

    return (left, top), (left + image_width * box['Width'], top + image_height * box['Height']) 


def image_information(photo): 

    #Get image width and height
    image = Image.open(open(photo,'rb'))
    width, height = image.size

    logger.info(f"Image information: {photo}")
    logger.info(f"Image Height: {height}")
    logger.info(f"Image Width: {width}") 
    
    return image


def annotate_image(rekog_resp, image, height, width, qf, max_faces):
    r = rekog_resp

    # call detect faces and show face age and placement
    # if found, preserve exif info
    stream = io.BytesIO()

    if 'exif' in image.info:
        exif = image.info['exif']
        image.save(stream, format=image.format, exif=exif)
    else:
        image.save(stream, format=image.format)

    if 'OrientationCorrection' in r:
        logger.info('Orientation: ' + r['OrientationCorrection'])
    else: 
        logger.info('No estimated orientation. Check Exif')    
    
    draw = ImageDraw.Draw(image)
    font = None

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    if "SearchedFaceBoundingBox" in r: # Search
        conf = r.get("SearchedFaceConfidence")
        
        draw.text((0, 0), f"quality: {qf} / max faces: {max_faces}", font=font, fill=(255, 255, 0, 128))
        x, y = show_bounding_box_positions(height, width, r['SearchedFaceBoundingBox'], "ROTATE_0")
        
        if conf >= SIMILARITY_THRESHOLD:
            draw.rectangle((x, y), fill=None, outline="#00ff00", width=8)
            draw.text(x, f"Conf: {conf}", font=font, fill=(255, 255, 0, 128))
        else:
            draw.rectangle((x, y), fill=None, outline="#ff0000", width=4)
            draw.text(x, f"Conf: {conf}", font=font, fill=(255, 255, 0, 128))
        
        logger.info("FaceModelVersion: {}".format(r.get("FaceModelVersion")))
    else: # Index
        draw.text((0, 0), f"quality: {qf} / max faces: {max_faces}", font=font, fill=(255, 255, 0, 128))

        for face_rec in rekog_resp.get("FaceRecords"):

            face = face_rec.get("Face") 
            face_id = face.get("FaceId")
            conf = face.get("Confidence")
            bb = face.get("BoundingBox")

            logger.info("\n=========")
            logger.info(f"face_id: {face_id}")
            logger.info("confidence: {}".format(face.get("Confidence")))
            logger.info("face_id: {}".format(json.dumps(face_rec, indent=4)))
            
            #if 'OrientationCorrection' in r:
            x, y = show_bounding_box_positions(height, width, face_rec['Face']['BoundingBox'], "ROTATE_0")
            
            if conf >= SIMILARITY_THRESHOLD:
                draw.rectangle((x, y), fill=None, outline="#00ff00", width=8)
                draw.text(x, f"Conf: {conf}", font=font, fill=(255, 255, 0, 128))
            else:
                draw.rectangle((x, y), fill=None, outline="#ff0000", width=4)
                draw.text(x, f"Conf: {conf}", font=font, fill=(255, 255, 0, 128))
        
    return image


def clean_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    
    return ''.join(c for c in filename if c in valid_chars)


def search_faces(coll_id, filename, qf="HIGH", max_faces=5, fmt=SIMILARITY_THRESHOLD):
    logger.info(f">search_faces file: '{filename}' / quality: {qf} / max faces: {max_faces}")
    
    image = image_information(filename)
    width, height = image.size

    blob_data = None

    with open(filename, 'rb') as file_t:
        blob_data = file_t.read()

    external_image_id = clean_filename(filename)

    logger.info(f">Calling rekog... {external_image_id}")
    start = time.time()
    r = rek_cli.search_faces_by_image(
        CollectionId=coll_id,
        Image={
            'Bytes': blob_data,
        },
        MaxFaces=max_faces,
        FaceMatchThreshold=fmt,
        QualityFilter=qf
    )
    end = time.time()
    logger.info("<Calling rekog FINISHED Elapsed time: {} ms".format((end - start) * 1000))

    logger.info(r)

    image = annotate_image(r, image, height, width, qf, max_faces)

    image.show()


def index_faces(coll_id, filename, qf="HIGH", max_faces=5):
    logger.info(f">index_faces file: '{filename}' / quality: {qf} / max faces: {max_faces}")
    
    image = image_information(filename)
    width, height = image.size

    blob_data = None

    with open(filename, 'rb') as file_t:
        blob_data = file_t.read()

    external_image_id = clean_filename(filename)

    logger.info(f">Calling rekog... {external_image_id}")
    start = time.time()
    r = rek_cli.index_faces(
        CollectionId=coll_id,
        Image={
            'Bytes': blob_data,
        },
        ExternalImageId=external_image_id,
        DetectionAttributes=[
            'ALL',
        ],
        MaxFaces=max_faces,
        QualityFilter=qf
    )
    end = time.time()
    logger.info("<Calling rekog FINISHED Elapsed time: {} ms".format((end - start) * 1000))

    # logger.info(r)

    image = annotate_image(r, image, height, width, qf, max_faces)
    
    image.show()

    id = uuid.uuid4()

    output_file = "{}-{}-{}.jpg".format(id, os.path.basename(filename), qf)

    image.save(output_file, "JPEG")

    logger.info(f"open {output_file}")

    # close image

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    
    return dst

def annotate_image_comparison(rekog_resp, image_src, image_tgt, image_src_height, image_src_width, qf, max_faces):
    r = rekog_resp

    # call detect faces and show face age and placement
    # if found, preserve exif info

    # image_concat = image_src
    image_concat = get_concat_h(image_src, image_tgt)
    height = image_concat.height
    width = image_concat.width

    draw = ImageDraw.Draw(image_concat)
    font = None

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    if "SourceImageFace" in r:
        face_src = r.get("SourceImageFace")
        conf = face_src.get("Confidence")
        
        annotate(image_concat, draw, (0, 0), f"quality: {qf} / max faces: {max_faces}", font)
        #draw.text((0, 0), f"quality: {qf} / max faces: {max_faces}", font=font, fill=(255, 255, 0, 128))
        x, y = show_bounding_box_positions(image_src.height, image_src.width, face_src.get("BoundingBox"), "ROTATE_0")
        adjusted_x = x
        adjusted_y = y

        # if conf >= SIMILARITY_THRESHOLD:
        draw.rectangle((adjusted_x, adjusted_y), fill=None, outline="#00ff00", width=8)
        # draw.text(adjusted_x, f"SOURCE\nIS FACE? {conf:.3f}", font=font, fill=(255, 255, 0, 128))
        annotate(image_concat, draw, adjusted_x, f"SOURCE IS FACE? {conf:.3f}", font)
    
    
    for fm in r.get("FaceMatches"):
        logger.info(fm)
        sim = fm.get("Similarity")
        
        face = fm.get("Face")        
        bb = face.get("BoundingBox")
        conf = face.get("Confidence")
        
        x, y = show_bounding_box_positions(image_tgt.height, image_tgt.width, bb, "ROTATE_0")
        adjusted_x = (x[0] + image_src.width, x[1])
        adjusted_y = (y[0] + image_src.width, y[1])

        draw.rectangle((adjusted_x, adjusted_y), fill=None, outline="#0000ff", width=4)
        #draw.text(adjusted_x, f"MATCH\nIS FACE? {conf:.3f}\nSIM: {sim:.3f}", font=font, fill=(255, 255, 0, 128))
        annotate(image_concat, draw, adjusted_x, f"MATCH / IS FACE? {conf:.3f} / SIM: {sim:.3f}", font)


    for uf in r.get("UnmatchedFaces"):
        logger.info(uf)
        bb = uf.get("BoundingBox")
        x, y = show_bounding_box_positions(image_tgt.height, image_tgt.width, bb, "ROTATE_0")
        adjusted_x = (x[0] + image_src.width, x[1])
        adjusted_y = (y[0] + image_src.width, y[1])

        print(bb)

        draw.rectangle((adjusted_x, adjusted_y), fill=None, outline="#ff0000", width=4)
        annotate(image_concat, draw, adjusted_x, f"UNMATCH / IS FACE? {conf:.3f}", font)
        # draw.text(adjusted_x, f"UNMATCH\nIS FACE? {conf:.3f}", font=font, fill=(255, 255, 0, 128))


    return image_concat 

def compare_faces(coll_id, filename_src, filename_tgt, qf="HIGH", max_faces=5, fmt=COMP_FACE_THRESHOLD):
    logger.info(f">compare_faces file: '{filename}' / quality: {qf} / max faces: {max_faces}")
    
    image_src = image_information(filename_src)
    width, height = image_src.size

    image_tgt = image_information(filename_tgt)
    width, height = image_tgt.size

    blob_data_src = None
    blob_data_tgt = None

    with open(filename_src, 'rb') as file_t:
        blob_data_src = file_t.read()

    with open(filename_tgt, 'rb') as file_t:
        blob_data_tgt = file_t.read()

    logger.debug(" compare_faces - blob_data_src size: '%d'", len(blob_data_src))
    logger.debug(" compare_faces - blob_data_tgt size: '%d'", len(blob_data_tgt))

    logger.info(f" compare_faces - calling rekog...")
    
    start = time.time()
    r = None 

    try: 
        r = rek_cli.compare_faces(
            SourceImage={
                'Bytes': blob_data_src
            },
            TargetImage={
                'Bytes': blob_data_tgt
            },
            SimilarityThreshold=fmt,
            QualityFilter=qf
        )
    except rek_cli.exceptions.InvalidParameterException as e:
        logger.warning(" compare_faces - check if there are faces on the images")
    except Exception as e:
        logger.error("unknown error", e)
        
    end = time.time()
    
    logger.info(" compare_faces - calling rekog FINISHED Elapsed time: {} ms".format((end - start) * 1000))

    if r:
        logger.info(r)        
        image_comp = annotate_image_comparison(r, image_src, image_tgt, height, width, qf, max_faces)
        image_comp.show()
    else:
        logger.error(" compare_faces - error processing")


if __name__ == "__main__":
    
    option = sys.argv[1]
    filename = sys.argv[2]
    filename_other = sys.argv[3] if len(sys.argv) > 3 else ""

    logger.info("ARH_COLL_ID: '%s'", COLL_ID)
    logger.info("CMD: '%s'", option)
    logger.info("FILE 1: '%s'", filename)
    logger.info("FILE 2: '%s'", filename_other)

    if option.lower() == "if":
        index_faces(COLL_ID, filename)
    elif option.lower() == "sf":
        search_faces(COLL_ID, filename)
    elif option.lower() == "cf":
        compare_faces(COLL_ID, filename, filename_other, qf="HIGH", max_faces=5, fmt=COMP_FACE_THRESHOLD)
    else:
        logger.info("Unknown option")

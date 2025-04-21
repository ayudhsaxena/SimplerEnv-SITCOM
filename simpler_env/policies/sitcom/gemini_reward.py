# gemini_reward.py

import os
import base64
import re
from io import BytesIO
from typing import List, Dict, Any, Union
from PIL import Image
import google.generativeai as genai
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCysQCaT_h-ZDTkV4Gnc3o7gSdECxcmChg")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── Helpers ───────────────────────────────────────────────────────────────────
def encode_image(path: str) -> str:
    """Read a file and return its base64‑PNG string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



def pil_to_base64(img: Union[Image.Image, np.ndarray]) -> str:
    """
    Encode a PIL image or numpy array into a base64‑PNG string.
    """
    if isinstance(img, np.ndarray):
        # Convert H×W×C uint8 array to PIL
        img = Image.fromarray(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─── In‑Context Examples ────────────────────────────────────────────────────────
_example_data = [
    {
        "image1": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/0_0_img1.png"),
        "image2": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/0_0_img2.png"),
        "reward": "1",
        "reason": (
            "The gripper in image 1 is positioned very close to the carrot, suggesting an attempt to interact with it. "
            "In image 2, the gripper has moved slightly towards the carrot, indicating only some progress. "
            "The reward of 2 reflects partial movement towards the goal without successfully grasping the object."
        )
    },
    {
        "image1": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/2_2_img1.png"),
        "image2": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/2_2_img2.png"),
        "reward": "2",
        "reason": (
            "In image 1, the gripper is precisely positioned above the carrot, successfully making contact and appearing to grasp it. "
            "By image 2, the gripper is picking the carrot off the surface, demonstrating controlled manipulation and task advancement. "
            "The reward of 4 reflects this significant progress."
        )
    },
    {
        "image1": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img1.png"),
        "image2": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img2.png"),
        "reward": "0",
        "reason": (
            "In both images, the gripper remains close to the carrot but shows no meaningful interaction or progress toward grasping it. "
            "The carrot’s position is unchanged, and the gripper appears static or only slightly adjusted. "
            "The reward of 0 reflects a inability of movement or incorrect movement toward the task goal."
        )
    },
    {
        "image1": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/4_7_img1.png"),
        "image2": encode_image("/data/user_data/rishisha/sitcom/trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/4_7_img2.png"),
        "reward": "1",
        "reason": (
            "In the first image, the gripper is hovering above the carrot without significant interaction. "
            "In the second image, the gripper has moved slightly closer to the carrot, showing a small step toward attempting a pick. "
            "The reward of 1 reflects this minor progress in positioning, even though the carrot has not been grasped yet."
        )
    },
]

# Build the static part of the contents list once
_static_contents: List[Dict[str, Any]] = []
_subtask = "pick up the carrot"

_system_instruction = (
    "You are a helpful reward model. Your task is to evaluate how well a transition from image1 to image2 "
    f"accomplishes the following subtask:\n\n**Subtask:** {_subtask}\n\n"
    "Provide both a <reason> explaining your evaluation, and a scalar <reward> (integer between 0 and 2) "
    "indicating how beneficial or detrimental the transition is for this subtask."
)

for idx, ex in enumerate(_example_data):
    # user turn
    prompt = _system_instruction if idx == 0 else "Evaluate the following transition.\n\n"
    prompt += (
        "The first image is <image1> (initial), the second is <image2> (result).\n"
        "Return both a <reason> and a single integer <reward> tag."
    )
    _static_contents.append({
        "role": "user",
        "parts": [
            prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(ex["image1"])},
            {"mime_type": "image/jpeg", "data": base64.b64decode(ex["image2"])},
        ]
    })
    # assistant turn
    _static_contents.append({
        "role": "model",
        "parts": [f"<reason>{ex['reason']}</reason>\n<reward>{ex['reward']}</reward>"]
    })


# ─── Stub for subtask extraction ───────────────────────────────────────────────
def extract_subtask(prev_img: Image.Image, cur_img: Image.Image) -> str:
    """
    Replace this with your real logic.
    """
    return _subtask

# ─── The Gem‑ini Reward Function ───────────────────────────────────────────────
def gemini_reward(prev_img: Image.Image, cur_img: Image.Image) -> int:
    """
    Evaluate a single transition using Gemini with in‑context examples.
    
    Args:
        prev_img: before‑action frame (PIL)
        cur_img:  after‑action frame  (PIL)
    
    Returns:
        reward ∈ {0,1,2}
    """
    # 1) determine subtask on the fly (if you want to override per example)
    subtask = extract_subtask(prev_img, cur_img)
    
    # 2) encode the query images
    img1_b64 = pil_to_base64(prev_img)
    img2_b64 = pil_to_base64(cur_img)
    
    # 3) build the full contents list: examples + our test query
    contents = list(_static_contents)  # shallow copy of the examples
    # append our query
    prompt = "Evaluate the following transition.\n\n" + (
        "The first image is <image1> (initial), the second is <image2> (result).\n"
        "Return both a <reason> and a single integer <reward> tag."
    )
    contents.append({
        "role": "user",
        "parts": [
            prompt,
            {"mime_type": "image/png", "data": base64.b64decode(img1_b64)},
            {"mime_type": "image/png", "data": base64.b64decode(img2_b64)},
        ]
    })
    
    # 4) call Gemini
    response = model.generate_content(contents=contents)
    text = response.text or ""
    
    # 5) parse out the reward
    m = re.search(r"<reward>(\d+)</reward>", text)
    if m:
        return int(m.group(1))
    # fallback
    return 0

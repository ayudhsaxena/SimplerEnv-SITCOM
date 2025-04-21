import os
import base64
import google.generativeai as genai

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

genai.configure(api_key="AIzaSyCysQCaT_h-ZDTkV4Gnc3o7gSdECxcmChg")
model = genai.GenerativeModel('gemini-2.0-flash')

example_data = [
    {
        "image1": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/0_0_img1.png"),
        "image2": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/0_0_img2.png"),
        "reward": "1",
        "reason": "The gripper in image 1 is positioned very close to the carrot, suggesting an attempt to interact with it. In image 2, the gripper has moved slightly towards the carrot, indicating only some progress. The reward of 2 reflects partial movement towards the goal without successfully grasping the object."
    },
    {
        "image1": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/2_2_img1.png"),
        "image2": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/2_2_img2.png"),
        "reward": "2",
        "reason": "In image 1, the gripper is precisely positioned above the carrot, successfully making contact and appearing to grasp it. By image 2, the gripper is picking the carrot off the surface, demonstrating controlled manipulation and task advancement. The reward of 4 reflects this significant progress."
    },
    {
        "image1": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img1.png"),
        "image2": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img2.png"),
        "reward": "0",
        "reason": "In both images, the gripper remains close to the carrot but shows no meaningful interaction or progress toward grasping it. The carrot’s position is unchanged, and the gripper appears static or only slightly adjusted. The reward of 0 reflects a inability of movement or incorrect movement toward the task goal."
    },
    {
        "image1": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/4_7_img1.png"),
        "image2": encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/4_7_img2.png"),
        "reward": "1",
        "reason": "In the first image, the gripper is hovering above the carrot without significant interaction. In the second image, the gripper has moved slightly closer to the carrot, showing a small step toward attempting a pick. The reward of 1 reflects this minor progress in positioning, even though the carrot has not been grasped yet."
    },
]
# 2
# test_image1 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/10_1_img1.png")
# test_image2 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/10_1_img2.png")

# 0
# test_image1 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img1.png")
# test_image2 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img2.png")

# 1
test_image1 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img1.png")
test_image2 = encode_image("trajectories_ft_sim/PutCarrotOnPlateInScene-v0/images/7_3_img2.png")

subtask = "pick up the carrot"

contents = []

system_instruction = (
    f"You are a helpful reward model. Your task is to evaluate how well a transition from image1 to image2 "
    f"accomplishes the following subtask:\n\n"
    f"**Subtask:** {subtask}\n\n"
    f"Provide both a <reason> explaining your evaluation, and a scalar <reward> (integer between 0 and 2) "
    f"indicating how beneficial or detrimental the transition is for this subtask. 0 means incorrect or unable movement towards the goal. 1 means movement towards the object. 2 means completing the goal."
)

# In-context examples
for idx, ex in enumerate(example_data):
    user_prompt = system_instruction if idx == 0 else "Evaluate the following transition.\n\n"

    user_prompt += (
        "The first image is <image1> (initial), the second is <image2> (result).\n"
        "Return both a <reason> and a single integer <reward> tag."
    )

    contents.append({
        "role": "user",
        "parts": [
            user_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(ex['image1'])},
            {"mime_type": "image/jpeg", "data": base64.b64decode(ex['image2'])}
        ]
    })
    contents.append({
        "role": "model",
        "parts": [f"<reason>{ex['reason']}</reason>\n<reward>{ex['reward']}</reward>"]
    })

# Test query
test_query_prompt = (
    "Evaluate the following transition.\n"
    "The first image is <image1> (initial), the second is <image2> (result).\n"
    "Return both a <reason> and a single integer <reward> tag."
)

contents.append({
    "role": "user",
    "parts": [
        test_query_prompt,
        {"mime_type": "image/jpeg", "data": base64.b64decode(test_image1)},
        {"mime_type": "image/jpeg", "data": base64.b64decode(test_image2)}
    ]
})

response = model.generate_content(contents=contents) 

print("response = ", response.text)

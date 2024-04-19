# src/prompt_definition.py

def define_prompts():
    """
    Define prompts for the chatbot.

    Returns:
    - list: List of prompts.
    """
    prompts = [
        "Can you provide details about a specific device?",
        "What are the specifications of a device with a given ID?",
        "Tell me about a device manufactured by a specific manufacturer.",
        "What operating systems are available?",
        "Can you list devices running a specific operating system?",
        "Give me details about devices running Windows 11.",
        "Tell me about devices with a certain amount of RAM.",
        "List devices with SSD storage.",
        "Give me information about devices with a particular processor.",
        "Which devices are located in a specific country?",
        "List devices located in a particular state.",
        "Tell me about devices in a specific city.",
        "Show me devices with warranties expiring by a certain date.",
        "Who is the owner of a device with a given serial number?",
        "List devices owned by a specific person."
    ]
    return prompts

if __name__ == "__main__":
    # Define prompts
    prompts = define_prompts()

    # Display prompts
    print("Prompts:")
    for prompt in prompts:
        print(prompt)

# Jayden's Small Language Model

Current model is 500m params. Core model and tokenizer rolled by hand, but default tokenizer is huggingface's. Default attention is also flashattention/pytorch fused kernels. Trained on a subset of smolLM-corpus, SFT for prompts on smoltalk2.

### Example

```
you> Hello there!
bot> Hello! It's nice to meet you. What brings you here today? Are you looking for advice, help with a specific issue, or just someone to chat with for company?
[13 prompt tokens]

you> What is the biggest national park in california
bot> Ah, yes, there's the Big Sur National Park in California. It's a huge area that's both a natural wonder and a treasure trove of wildlife and adventure. It's known for its rugged coastline, diverse ecosystems, and majestic waterfalls. If you're in the right spirit, you can spend hours exploring the park's many trails, swimming in its many hot springs, and taking in its stunning natural beauty.

One of the main attractions is the Quinault Indian Village, which is situated on the mainland and offers a glimpse into the lives of Native American tribes that have lived here for centuries. It's a great place to learn about the region's rich cultural history and to enjoy a traditional Native American feast at the village.

Another highlight is the Anastasia Palisades Lodge, the largest public lodge in the state. It's an impressive venue for events like outdoor concerts, raffles, and art exhibits. And for those who love the thrill of hiking, the 263-acre Parkview Mountain State Park is a must-visit for its breathtaking views of the Bay Area.

Lastly, if you're looking for something a bit more off the beaten path, the Muir Woods National Monument in California is another incredible destination. It's a vast wilderness area with some of the most iconic and stunning natural features in the world, including giant sequoia groves, gorges, waterfalls, and volcanic landscapes.

All of these spots combine to make the Big Sur National Park a truly unforgettable experience. Would you like me to suggest a specific itinerary for you to explore on your next trip?
[69 prompt tokens]

you> write me a short story about a man who really likes cake
bot> In a quiet suburban home on the outskirts of the city, a 42-year-old man named Malcolm stood before his grand box of chocolate-coated Arbequins, carefully stacking them in a single layer. He had heard tales of these iconic French-style sugar cookies, but since he had never taken a bite, he was eager to experience their flavor for himself.

As he worked, his face illuminated with a warm, golden light, he was transported to a bygone era when French patisseries were the epitome of culinary sophistication. He loved the way the delicate, edible centers were adorned with intricate sugar flowers and gold leaf, and the rich, buttery flavor was like nothing he had ever tasted.

But beneath the exterior, a secret was brewing. Malcolm was secretly an artist, and he had always been fascinated by the way the delicate layers of sugar and chocolate blended together to create something truly magical. He began to experiment with his own creations, using techniques he had learned from his grandmother, who was an accomplished cake maker herself.

One day, while working on a particularly complex dessert, Malcolm's apartment suddenly felt like it was being stirred. He felt a chill run down his spine, and his mind, like an oven, was on high alert. His pastry chef, Marie, and her husband, Pierre, were trapped in a frantic race to get him off the sofa, but it was too late.

Malcolm's imagination took over, and he started to envision a perfect wedding cake, adorned with gold leaf and a vibrant flower. But just as he was about to pitch it to his lawyer to create the perfect marriage proposal, he heard a loud, sudden noise outside. He went to investigate and found a woman frantically standing on the steps of his apartment, her phone clutched in her hand.

With a mix of fear and hope, Malcolm sprang into action. He saw that the woman was a product of his art class, and in her case, it was a challenge. Without hesitation, he reached for the remaining ingredients and began to work the fragile cake on a few key design elements, using a napkin to serve as a makeshift plate.

As the final layer of the cake was added, Malcolm's hands trembled with anticipation, and for the first time in years, he felt a sense of purpose. He knew that this was his moment to create something truly beautiful, something that would capture the hearts of all who loved to eat. And
[426 prompt tokens]
```

It does hallucinate a lot.

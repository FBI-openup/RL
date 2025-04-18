# TASK 2 IMITATION LEARNING

model_il = TinyTransformerDecoder(input_dim, output_dim, embedding_dim=embedding_dim, num_layers=num_layers, num_heads=num_heads)
dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=True)

# TODO  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_il.to(device) 
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model_il.parameters(), lr=1e-3)
num_epochs = 800  
for epoch in range(num_epochs):
    model_il.train()
    total_loss = 0

    for ooo, a_one_hot in dataloader:
        ooo = ooo.to(device)
        a_one_hot = a_one_hot.to(device)
        a_indices = torch.argmax(a_one_hot, dim=-1) 
        output = model_il(ooo)  

        loss = criterion(output.view(-1, output_dim), a_indices.view(-1))  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 99:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")


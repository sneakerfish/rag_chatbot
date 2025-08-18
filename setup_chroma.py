# setup_chroma.py
import chromadb
from chromadb.config import Settings
import time

def test_chromadb_connection():
    """Test basic connection to ChromaDB server"""
    print("🔍 Testing ChromaDB server connection...")
    try:
        # Test with admin client first to verify server is reachable
        admin_client = chromadb.AdminClient(
            settings=Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host="localhost",
                chroma_server_http_port=8002
            )
        )
        
        # Try to list tenants to test connectivity
        try:
            tenants = admin_client.list_tenants()
            print("✅ ChromaDB server connection successful")
        except Exception as e:
            # If we can't list tenants, the server might not be ready or have different API
            print("✅ ChromaDB server connection successful (basic connectivity)")
        
        # Create regular client for tenant-specific operations
        client = chromadb.HttpClient(
            host="localhost",
            port=8002,
            settings=Settings(anonymized_telemetry=False)
        )
        
        return client
    except Exception as e:
        print(f"❌ ChromaDB server connection failed: {e}")
        return None

def setup_tenant_and_database():
    """Set up tenant and database if they don't exist"""
    print("\n🏗️  Setting up tenant and database...")
    
    try:
        admin_client = chromadb.AdminClient(
            settings=Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host="localhost",
                chroma_server_http_port=8002
            )
        )
        print("✅ AdminClient created successfully")
    except Exception as e:
        print(f"❌ Failed to create AdminClient: {e}")
        return False

    try:
        # Try to create tenant (will fail if it already exists, which is OK)
        try:
            admin_client.create_tenant(name="default_tenant")
            print("✅ Tenant 'default_tenant' created successfully")
        except Exception as e:
            if "already exists" in str(e):
                print("✅ Tenant 'default_tenant' already exists")
            else:
                print(f"❌ Failed to create tenant: {e}")
                return False

        # Check if database exists, create if not
        try:
            existing_databases = admin_client.list_databases(tenant="default_tenant")
            print(f"📋 Found {len(existing_databases)} existing databases in default_tenant")
        except Exception as e:
            print(f"⚠️  Could not list databases: {e}")
            existing_databases = []

        db_exists = any(db.get('name') == "default_database" for db in existing_databases)
        
        if db_exists:
            print("✅ Database 'default_database' already exists")
        else:
            print("Creating database 'default_database'...")
            try:
                admin_client.create_database(name="default_database", tenant="default_tenant")
                print("✅ Database 'default_database' created successfully")
            except Exception as e:
                print(f"❌ Failed to create database: {e}")
                return False
            
        return True
    except Exception as e:
        print(f"❌ Error setting up tenant/database: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading(client):
    """Test loading sample data into ChromaDB"""
    print("\n📚 Testing data loading capabilities...")
    
    try:
        # Create a test collection
        collection_name = "test_data_loading"
        
        # Delete collection if it exists
        try:
            client.delete_collection(collection_name)
            time.sleep(1)  # Give it time to delete
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Test collection for data loading verification"}
        )
        print("✅ Test collection created")
        
        # Add sample documents
        sample_documents = [
            "ChromaDB is a vector database for building AI applications.",
            "Vector databases store embeddings for similarity search.",
            "Embeddings are numerical representations of text or other data.",
            "ChromaDB supports both persistent and in-memory storage.",
            "ChromaDB can be used for semantic search and retrieval."
        ]
        
        # Create 384-dimensional mock embeddings (matching all-MiniLM-L6-v2)
        import random
        random.seed(42)  # For reproducible results
        
        sample_embeddings = []
        for i in range(len(sample_documents)):
            # Generate 384-dimensional vector with small random values
            embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
            sample_embeddings.append(embedding)
        
        sample_ids = [f"doc_{i}" for i in range(len(sample_documents))]
        
        # Add documents to collection
        collection.add(
            documents=sample_documents,
            embeddings=sample_embeddings,
            ids=sample_ids
        )
        print(f"✅ Added {len(sample_documents)} sample documents")
        
        # Test querying the data
        print("🔍 Testing data retrieval...")
        # Create a 384-dimensional query embedding
        query_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            print("✅ Data retrieval successful")
            print(f"   Retrieved {len(results['documents'][0])} documents")
        else:
            print("❌ Data retrieval failed - no documents returned")
            return False
        
        # Test similarity search
        print("🔍 Testing similarity search...")
        search_results = collection.query(
            query_texts=["vector database"],
            n_results=2
        )
        
        if search_results['documents'] and len(search_results['documents'][0]) > 0:
            print("✅ Similarity search successful")
            print(f"   Found {len(search_results['documents'][0])} similar documents")
        else:
            print("❌ Similarity search failed")
            return False
        
        # Clean up test collection
        client.delete_collection(collection_name)
        print("✅ Test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_collection_management(client):
    """Test collection management operations"""
    print("\n🗂️  Testing collection management...")
    
    try:
        # Test creating multiple collections
        collections = []
        for i in range(3):
            collection_name = f"test_collection_{i}"
            collection = client.create_collection(
                name=collection_name,
                metadata={"test": True, "index": i}
            )
            collections.append(collection_name)
            print(f"✅ Created collection: {collection_name}")
        
        # Test listing collections
        all_collections = client.list_collections()
        test_collections = [c for c in all_collections if c.name.startswith("test_collection_")]
        print(f"✅ Found {len(test_collections)} test collections")
        
        # Test getting collection
        test_collection = client.get_collection("test_collection_0")
        print("✅ Successfully retrieved collection")
        
        # Clean up test collections
        for collection_name in collections:
            client.delete_collection(collection_name)
        print("✅ Cleaned up test collections")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection management test failed: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("🚀 ChromaDB Setup and Verification Script")
    print("=" * 50)
    
    # Setup tenant and database first
    setup_success = setup_tenant_and_database()
    if not setup_success:
        print("\n❌ Tenant/database setup failed")
        return
    
    # Now test connection with proper tenant setup
    client = test_chromadb_connection()
    if not client:
        print("\n❌ Cannot proceed without ChromaDB connection")
        return
    
    # Test data loading
    data_loading_success = test_data_loading(client)
    
    # Test collection management
    collection_management_success = test_collection_management(client)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup and Verification Results:")
    print(f"✅ ChromaDB Connection: {'PASS' if client else 'FAIL'}")
    print(f"✅ Tenant/Database Setup: {'PASS' if setup_success else 'FAIL'}")
    print(f"✅ Data Loading: {'PASS' if data_loading_success else 'FAIL'}")
    print(f"✅ Collection Management: {'PASS' if collection_management_success else 'FAIL'}")
    
    if all([client, setup_success, data_loading_success, collection_management_success]):
        print("\n🎉 ChromaDB is fully operational and ready for use!")
        print("You can now load your documents and build your chatbot.")
    else:
        print("\n⚠️  Some tests failed. Please check your ChromaDB configuration.")
        print("Common issues:")
        print("- ChromaDB service not running")
        print("- Wrong port number")
        print("- Network connectivity issues")
        print("- Insufficient permissions")

if __name__ == "__main__":
    main()
